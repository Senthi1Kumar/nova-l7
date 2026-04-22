"""
NOVA Layer 7 — LLM-based Intent Classifier.

Drop-in for IntentClassifier that uses an OpenRouter tool-calling LLM to map
driver utterances → {intent, entities}. Tool definitions mirror Nova's DM
handler contracts (handle_vehicle_control, handle_navigation, handle_media,
handle_order_flow, handle_communication), so the downstream DM/FSM does not
change.

Compound utterances are handled natively: the LLM emits one tool call per
distinct intent, which this classifier returns as a list[IntentResult]. The
DM's intent_queue drains them in priority order — same path as the regex path.

Feature-flagged via NOVA_DM_MODE=tool_calling. Falls back to regex
IntentClassifier on any failure (no API key, timeout, parse error).

Env:
  OPENROUTER_API_KEY          required
  NOVA_DM_LLM_MODEL           default: google/gemma-4-26b-a4b-it
  NOVA_DM_LLM_TIMEOUT_S       default: 3.0
"""

import json
import logging
import os
import re
import time
from typing import Optional

from intent_classifier import (
    IntentClassifier,
    IntentResult,
    extract_entities,
)

logger = logging.getLogger("LLMIntentClassifier")

# ── Tool schemas — shapes match DM handler entity contracts ───────────────────

NOVA_INTENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vehicle_control",
            "description": (
                "Control a vehicle component. Use for explicit commands like "
                "'turn on the AC', 'open the sunroof', AND implicit comfort "
                "phrases like 'I'm cold' (→ heater on), 'it's stuffy' (→ ac on). "
                "For compound vehicle commands ('AC on and sunroof open'), emit "
                "one tool call per component."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": [
                            "ac", "heater", "fan", "window", "sunroof",
                            "lights", "seat", "wiper", "horn",
                            "rear_camera", "defrost",
                        ],
                    },
                    "action": {"type": "string", "enum": ["on", "off"]},
                    "temperature_c": {
                        "type": "integer",
                        "description": "Target Celsius, if the driver specified a number.",
                    },
                    "fan_speed": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["component", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigation",
            "description": (
                "Start navigation. Use for 'take me to', 'go to', 'navigate to', "
                "'I'm headed to', 'drive to', 'directions to', 'route to'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Destination as spoken ('office', 'home', 'nearest Starbucks').",
                    },
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Play / pause / skip music, podcasts, or playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to play (song/artist/album). Omit for pause/skip.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "skip", "next", "previous"],
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Order food, coffee, or goods. Emit ONE tool call per item. "
                "For compound orders ('two coffees and three pizzas') emit two "
                "separate calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "What to order (e.g., 'coffee', 'pizza', 'latte').",
                    },
                    "quantity": {
                        "type": "integer", "minimum": 1, "maximum": 10,
                    },
                    "size": {
                        "type": "string",
                        "enum": [
                            "small", "medium", "large", "grande",
                            "venti", "tall", "regular", "extra large",
                        ],
                    },
                    "merchant": {
                        "type": "string",
                        "description": "Merchant if named (Starbucks, Domino's, etc.).",
                    },
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "communication",
            "description": "Call, text, WhatsApp, or email a contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["call", "text", "whatsapp", "email"],
                    },
                    "contact": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel", "contact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_question",
            "description": (
                "User is asking a general question (weather, news, facts, "
                "small-talk). Use this when none of the specialised tools fit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The driver's question, verbatim.",
                    },
                },
                "required": ["text"],
            },
        },
    },
]


_SYSTEM_PROMPT = (
    "You are the intent router for Nova, an AI voice assistant in an EV. "
    "For every driver utterance, emit tool calls — one per distinct intent. "
    "Compound utterances (e.g., 'turn on AC and navigate to office') MUST "
    "produce multiple tool calls.\n\n"
    "Rules:\n"
    "- Use vehicle_control for any climate / cabin / lights / window / sunroof "
    "command, including implicit comfort phrases ('I'm cold', 'stuffy').\n"
    "- Use navigation for any 'go to X' / 'take me to X' / 'headed to X' form.\n"
    "- Use place_order for any 'order', 'buy', 'get me' food/drink request. "
    "ONE call per item.\n"
    "- Use general_question only when nothing else fits.\n"
    "- Never reply with plain text — always a tool call. If the utterance is "
    "truly empty or unintelligible, call general_question with the raw text."
)


# ── Fast-path keywords — never worth a round trip ─────────────────────────────

_STOP_KEYWORDS = ("stop talking", "shut up", "cancel", "nevermind",
                  "never mind", "forget it", "abort", "quiet")
_EMERGENCY_KEYWORDS = ("help me", "emergency", "call 911", "accident",
                       "crash", "heart attack")


class LLMIntentClassifier:
    """Drop-in replacement for IntentClassifier."""

    def __init__(self) -> None:
        self._model = os.getenv("NOVA_DM_LLM_MODEL", "google/gemma-4-26b-a4b-it")
        self._timeout = float(os.getenv("NOVA_DM_LLM_TIMEOUT_S", "3.0"))

        self._client = None
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    timeout=self._timeout,
                )
                logger.info(f"LLMIntentClassifier ready (model={self._model}, timeout={self._timeout}s)")
            except Exception as e:
                logger.error(f"OpenRouter init failed: {e}")

        # Fallback regex classifier — used on timeout/parse failure or when
        # no API key is configured.
        try:
            self._fallback = IntentClassifier()
            logger.info("Regex fallback classifier loaded.")
        except Exception as e:
            logger.error(f"Regex fallback init failed: {e}")
            self._fallback = None

    # ── Public API (mirror of IntentClassifier) ──────────────────────────────

    def classify(self, text: str) -> IntentResult:
        results = self.split_and_classify(text)
        return results[0] if results else IntentResult(
            intent="unknown", confidence=0.0, entities={}, raw_text=text
        )

    def split_and_classify(self, text: str) -> list:
        if not text or not text.strip():
            return [IntentResult(intent="unknown", confidence=0.0,
                                 entities={}, raw_text=text)]

        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Safety fast-path — never defer these to the LLM.
        if any(kw in text_lower for kw in _EMERGENCY_KEYWORDS):
            return [IntentResult(intent="emergency", confidence=1.0,
                                 entities={}, raw_text=text_clean)]
        if any(kw in text_lower for kw in _STOP_KEYWORDS):
            return [IntentResult(intent="stop", confidence=1.0,
                                 entities={}, raw_text=text_clean)]

        if self._client is None:
            return self._fallback_classify(text_clean)

        try:
            return self._llm_classify(text_clean)
        except Exception as e:
            logger.warning(f"LLM classify failed ({e}); falling back to regex.")
            return self._fallback_classify(text_clean)

    # ── Internals ────────────────────────────────────────────────────────────

    def _llm_classify(self, text: str) -> list:
        t0 = time.time()
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=NOVA_INTENT_TOOLS,
            tool_choice="required",
            temperature=0.0,
            max_tokens=256,
        )
        latency_ms = int((time.time() - t0) * 1000)

        choice = resp.choices[0] if resp.choices else None
        tool_calls = (choice.message.tool_calls or []) if choice else []

        if not tool_calls:
            logger.info(f"LLM returned no tool calls ({latency_ms}ms) — treating as general_question.")
            return [IntentResult(intent="general_question", confidence=0.5,
                                 entities={}, raw_text=text)]

        results: list[IntentResult] = []
        names: list[str] = []
        for tc in tool_calls:
            fn = tc.function
            name = fn.name or ""
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON in tool args: {fn.arguments!r}")
                continue

            ir = self._tool_to_intent(name, args, text)
            if ir is not None:
                results.append(ir)
                names.append(name)

        if not results:
            return [IntentResult(intent="general_question", confidence=0.5,
                                 entities={}, raw_text=text)]

        logger.info(f"LLM classified in {latency_ms}ms → {names}")
        return results

    def _tool_to_intent(self, name: str, args: dict, raw_text: str) -> Optional[IntentResult]:
        """Map one tool call → one IntentResult. Entity shapes match what the
        DM handlers already consume."""
        if name == "vehicle_control":
            entities: dict = {
                "component": args.get("component"),
                "action": args.get("action"),
            }
            if args.get("temperature_c") is not None:
                temp_c = int(args["temperature_c"])
                entities["temperature_c"] = temp_c
                entities["ac_temp_range"] = (
                    "cold" if temp_c <= 18 else "mid" if temp_c <= 22 else "hot"
                )
                if not entities.get("component"):
                    entities["component"] = "ac"
                    entities["action"] = entities.get("action") or "on"
            if args.get("fan_speed"):
                entities["fan_speed"] = args["fan_speed"]
            return IntentResult(
                intent="vehicle_control", confidence=0.95,
                entities=entities, raw_text=raw_text,
            )

        if name == "navigation":
            dest = (args.get("destination") or "").strip().rstrip(".?!")
            return IntentResult(
                intent="navigation", confidence=0.95,
                entities={"destination": dest or None},
                raw_text=raw_text,
            )

        if name == "media_control":
            entities = {}
            if args.get("query"):
                entities["query"] = args["query"]
            if args.get("action"):
                entities["action"] = args["action"]
            return IntentResult(
                intent="media", confidence=0.9,
                entities=entities, raw_text=raw_text,
            )

        if name == "place_order":
            # Reuse regex entity extractor for size/merchant/amount parsing
            # when the LLM omitted them — cheap insurance against sparse outputs.
            fallback = extract_entities(raw_text, "payment")
            entities = {
                "item": args.get("item") or fallback.get("item"),
                "quantity": int(args.get("quantity") or fallback.get("quantity") or 1),
            }
            if args.get("size"):
                entities["size"] = args["size"]
            elif fallback.get("size"):
                entities["size"] = fallback["size"]
            if args.get("merchant"):
                entities["merchant"] = args["merchant"].lower()
            elif fallback.get("merchant"):
                entities["merchant"] = fallback["merchant"]
            if fallback.get("amount"):
                entities["amount"] = fallback["amount"]
            if entities.get("merchant") and entities.get("item"):
                entities["query"] = f"{entities['item']} from {entities['merchant']}"
            elif entities.get("merchant"):
                entities["query"] = entities["merchant"]
            return IntentResult(
                intent="payment", confidence=0.9,
                entities=entities, raw_text=raw_text,
            )

        if name == "communication":
            entities = {
                "channel": args.get("channel"),
                "contact": args.get("contact"),
            }
            if args.get("message"):
                entities["message"] = args["message"]
            return IntentResult(
                intent="communication", confidence=0.9,
                entities=entities, raw_text=raw_text,
            )

        if name == "general_question":
            return IntentResult(
                intent="general_question", confidence=0.8,
                entities={}, raw_text=args.get("text") or raw_text,
            )

        logger.warning(f"Unknown tool name from LLM: {name!r}")
        return None

    def _fallback_classify(self, text: str) -> list:
        if self._fallback is None:
            return [IntentResult(intent="general_question", confidence=0.5,
                                 entities={}, raw_text=text)]
        return self._fallback.split_and_classify(text)
