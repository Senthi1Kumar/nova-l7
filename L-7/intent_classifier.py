"""
NOVA Layer 7 - Intent Classifier (Moonshine Voice)
Replaces regex-based classification with Gemma-300M semantic embeddings.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger("IntentClassifier")

# ── Compound intent detection ─────────────────────────────────────────────────

@dataclass
class CompoundCheck:
    is_compound: bool
    categories:  list   # list[str] — intent categories detected
    segments:    list   # list[str] — populated by CompoundSplitter, empty here


COMPOUND_SIGNALS: dict = {
    "vehicle_control": ["ac", "heater", "fan", "window", "sunroof", "lights",
                        "seat", "wiper", "cold", "hot", "warm", "stuffy"],
    "payment":         ["order", "coffee", "burger", "pizza", "buy", "want",
                        "get me", "latte", "food"],
    "navigation":      ["navigate", "go to", "take me", "directions", "route"],
    "media":           ["play", "music", "song", "playlist", "pause", "skip"],
    "communication":   ["call", "text", "message", "email", "send", "whatsapp"],
    "general_question":["what", "who", "how", "news", "weather", "tell me"],
}


class CompoundDetector:
    """Fast keyword scan — runs before any ML classification (~0.1ms)."""

    def check(self, text: str) -> CompoundCheck:
        text_lower = text.lower()
        detected = [
            cat for cat, signals in COMPOUND_SIGNALS.items()
            if any(s in text_lower for s in signals)
        ]
        return CompoundCheck(
            is_compound=len(detected) >= 2,
            categories=detected,
            segments=[],
        )


class CompoundSplitter:
    """
    Splits a compound utterance into segments at natural boundaries.
    Delimiters are tried cumulatively — longest first to avoid splitting
    inside single-intent clauses (e.g. "turn on AC and heater").
    """
    _DELIMITERS = [", ", ". ", " and also ", " then ", " also ", " and "]

    def split(self, text: str, max_segments: int = 5) -> list:
        segments = [text]
        for delim in self._DELIMITERS:
            new_segs = []
            for seg in segments:
                parts = re.split(re.escape(delim), seg, flags=re.IGNORECASE)
                new_segs.extend(p.strip() for p in parts if p.strip())
            segments = new_segs
            if len(segments) >= max_segments:
                break
        return segments[:max_segments]


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: str          # what type of command
    confidence: float    # how sure we are (0 to 1)
    entities: dict       # extracted info from command
    raw_text: str        # original command

# ── Entity extractors (Regex for slot filling is still useful here) ────────

KNOWN_MERCHANTS = [
    "starbucks", "subway", "mcdonalds", "mcdonald's",
    "pizza hut", "dominos", "domino's", "kfc",
    "blue bottle", "dunkin", "tim hortons",
    "chipotle", "panda express", "shell", "bp"
]

KNOWN_ITEMS = [
    "frappuccino", "caramel frappuccino", "mocha frappuccino",
    "latte", "caramel latte", "vanilla latte",
    "cappuccino", "espresso", "cold brew", "americano",
    "flat white", "macchiato", "chai latte",
    "veggie delight", "chicken teriyaki", "footlong",
    "big mac", "whopper", "double double",
    "sandwich", "burger", "pizza", "sub", "wrap",
    "coffee", "tea", "juice", "smoothie"
]

def extract_entities(text: str, intent: str) -> dict:
    entities  = {}
    text_lower = text.lower()

    if intent == "navigation":
        dest_match = re.search(
            r"(?:go to|navigate to|take me to|get me to|drive to|head to|route to|directions? to)\s+(.+)",
            text_lower
        )
        if dest_match:
            entities["destination"] = dest_match.group(1).strip()
        else:
            entities["destination"] = None

    if intent == "vehicle_control":
        controls = ["ac", "window", "sunroof", "heater", "fan",
                    "lights", "light", "seat", "wiper", "horn", "defrost",
                    "camera", "rear camera", "rear_camera"]
        component_aliases = {"roof": "sunroof", "a/c": "ac", "air conditioning": "ac",
                             "air conditioner": "ac", "air condition": "ac", "temperature": "ac",
                             "sound roof": "sunroof", "sun roof": "sunroof", "dac": "ac",
                             "camera": "rear_camera", "rear camera": "rear_camera",
                             "backup camera": "rear_camera", "back camera": "rear_camera",
                             "reverse camera": "rear_camera"}

        # Split on " and " to handle compound commands like
        # "switch off the AC and open the sunroof"
        clauses = [c.strip() for c in re.split(r'\band\b', text_lower) if c.strip()]

        all_commands = []
        last_action = None  # carry forward action across compound clauses
        for clause in clauses:
            comp = None
            for control in controls:
                if control in clause:
                    comp = control
                    break
            if not comp:
                for alias, canonical in component_aliases.items():
                    if alias in clause:
                        comp = canonical
                        break
            if not comp:
                continue

            action = None
            if any(w in clause for w in ["off", "close", "decrease", "lower", "down"]):
                action = "off"
            elif any(w in clause for w in ["on", "open", "increase", "higher", "up",
                                           "switch", "turn"]):
                action = "on"
            else:
                action = last_action or "on"  # inherit from previous clause
            last_action = action
            all_commands.append({"component": comp, "action": action})

        if all_commands:
            # Single command — flat entities for backward compat
            entities["component"] = all_commands[0]["component"]
            entities["action"] = all_commands[0]["action"]
            if len(all_commands) > 1:
                entities["commands"] = all_commands

        # Temperature extraction: "set it to 10c", "20 degrees"
        temp_match = re.search(r"(\d+)\s*(?:c\b|°c|degrees?(?:\s+celsius)?)", text_lower)
        if temp_match:
            temp_c = int(temp_match.group(1))
            entities["temperature_c"] = temp_c
            if temp_c <= 18:
                entities["ac_temp_range"] = "cold"    # ≤18°C max AC / ice cold
            elif temp_c <= 22:
                entities["ac_temp_range"] = "mid"     # 19–22°C comfort zone
            else:
                entities["ac_temp_range"] = "hot"     # ≥23°C warm/heat territory
        # If temperature extracted but no component resolved, default to AC
        if entities.get("temperature_c") and not entities.get("component"):
            entities["component"] = "ac"
            if not entities.get("action"):
                entities["action"] = "on"

        # Fan speed extraction
        if re.search(r"(?:fan|air).*?(?:to\s+)?(low|slow)\b", text_lower):
            entities["fan_speed"] = "low"
        elif re.search(r"(?:fan|air).*?(?:to\s+)?(medium|mid|moderate)\b", text_lower):
            entities["fan_speed"] = "medium"
        elif re.search(r"(?:fan|air).*?(?:to\s+)?(high|fast|full|max)\b", text_lower):
            entities["fan_speed"] = "high"
        elif re.search(r"(?:increase|raise|boost|turn up).*?(?:the\s+)?(?:fan|air)", text_lower):
            entities["fan_speed"] = "high"
        elif re.search(r"(?:decrease|lower|reduce|turn down).*?(?:the\s+)?(?:fan|air)", text_lower):
            entities["fan_speed"] = "low"

    if intent == "media":
        play_match = re.search(r"play\s+(.+)", text_lower)
        if play_match:
            entities["query"] = play_match.group(1).strip()

    if intent == "payment":
        sizes = ["small", "medium", "large", "grande", "venti", "tall", "regular", "extra large"]
        for size in sizes:
            if size in text_lower:
                entities["size"] = size
                break
                
        quantity_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a": 1, "an": 1}
        qty_match = re.search(r"\b(one|two|three|four|five|a|an|\d+)\b", text_lower)
        if qty_match:
            word = qty_match.group(1)
            raw_qty = quantity_map.get(word, int(word) if word.isdigit() else 1)
            entities["quantity"] = min(raw_qty, 10)  # Cap at 10 to prevent STT mishearing "100 ml" etc.
        else:
            entities["quantity"] = 1

        for merchant in KNOWN_MERCHANTS:
            if merchant in text_lower:
                entities["merchant"] = merchant
                break

        for item in sorted(KNOWN_ITEMS, key=len, reverse=True):
            if item in text_lower:
                entities["item"] = item
                break

        if not entities.get("item"):
            item_match = re.search(
                r"(?:order|get me|buy|i want|i'd like|give me)\s+(?:a\s+|an\s+)?(.+?)(?:\s+from|\s+at|\s+near|$)",
                text_lower
            )
            if item_match:
                entities["item"] = item_match.group(1).strip()

        # Compound order detection: "2 coffees and 2 pizzas"
        compound_clauses = [c.strip() for c in re.split(r'\band\b', text_lower) if c.strip()]
        if len(compound_clauses) > 1:
            multi_items = []
            for clause in compound_clauses:
                qty = 1
                qty_m = re.search(r"\b(one|two|three|four|five|a|an|\d+)\b", clause)
                if qty_m:
                    w = qty_m.group(1)
                    qty = quantity_map.get(w, int(w) if w.isdigit() else 1)
                item_found = None
                for it in sorted(KNOWN_ITEMS, key=len, reverse=True):
                    if it in clause:
                        item_found = it
                        break
                if item_found:
                    multi_items.append({"item": item_found, "quantity": min(qty, 10)})
            if len(multi_items) > 1:
                entities["multi_items"] = multi_items
                # Override with first compound item
                entities["item"]     = multi_items[0]["item"]
                entities["quantity"] = multi_items[0]["quantity"]

        dollar_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text_lower)
        if dollar_match:
            entities["amount"] = dollar_match.group(1)

        if not entities.get("amount"):
            rupee_match = re.search(
                r"(?:rs\.?|rupees?|inr)?\s*(\d+)", text_lower
            )
            if rupee_match:
                entities["amount"] = rupee_match.group(1)

        if entities.get("merchant") and entities.get("item"):
            entities["query"] = f"{entities['item']} from {entities['merchant']}"
        elif entities.get("merchant"):
            entities["query"] = entities["merchant"]
        elif entities.get("item"):
            entities["query"] = entities["item"]

    if intent == "communication":
        call_match = re.search(
            r"(?:call|phone|text|message|dial|ring)\s+(.+)", text_lower
        )
        if call_match:
            entities["contact"] = call_match.group(1).strip()

    return entities

# ── Main classifier ─────────────────────────────────────────────────────────

class IntentClassifier:
    def __init__(self):
        try:
            from moonshine_voice import IntentRecognizer, get_embedding_model
            from moonshine_voice.intent_recognizer import IntentMatch
            
            logger.info("Initializing Moonshine Intent Recognizer (gemma-300m)...")
            embedding_model_path, embedding_model_arch = get_embedding_model("embeddinggemma-300m", "q4")
            
            self.recognizer = IntentRecognizer(
                model_path=embedding_model_path,
                model_arch=embedding_model_arch,
                model_variant="q4",
                threshold=0.55  # Slightly lowered to catch natural phrasing
            )
            
            # Map canonical trigger phrases back to our broader intent categories
            self.trigger_to_intent = {
                "stop talking": "stop",
                "cancel order": "stop",
                "shut up": "stop",
                
                "turn on the AC": "vehicle_control",
                "turn off the lights": "vehicle_control",
                "roll down the windows": "vehicle_control",
                "adjust the fan": "vehicle_control",
                "switch on the heater": "vehicle_control",
                "turn on the heater": "vehicle_control",
                "open the sunroof": "vehicle_control",
                "turn on the seat heater": "vehicle_control",
                
                "navigate to the airport": "navigation",
                "take me home": "navigation",
                "give me directions": "navigation",
                "where is the nearest": "navigation",
                
                "play some music": "media",
                "pause the song": "media",
                "skip this track": "media",
                
                "order a coffee": "payment",
                "buy a burger": "payment",
                "pay for parking": "payment",
                "i want to order food": "payment",
                
                "call mom": "communication",
                "send a text message": "communication",
                "dial this number": "communication",
                
                "what is the weather like": "general_question",
                "tell me the news": "general_question",
                "what time is it": "general_question",
                "who is the president": "general_question"
            }
            
            for trigger in self.trigger_to_intent.keys():
                # We register them with a no-op handler because we will capture the intent via set_on_intent
                self.recognizer.register_intent(trigger, lambda t, u, s: None)
                
            self.latest_match = None
            def on_match(match: IntentMatch):
                self.latest_match = match
                
            self.recognizer.set_on_intent(on_match)
            logger.info("Moonshine Intent Recognizer ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Moonshine IntentRecognizer: {e}")
            self.recognizer = None

        self._detector = CompoundDetector()
        self._splitter = CompoundSplitter()

    def split_and_classify(self, text: str) -> list:
        """
        Returns a list of IntentResult.
        Single-intent utterances return a one-element list (zero overhead).
        Compound utterances are split, each segment classified, and unknown
        segments are silently dropped.
        """
        check = self._detector.check(text)
        if not check.is_compound:
            return [self.classify(text)]

        segments = self._splitter.split(text, max_segments=5)
        results = [
            self.classify(seg)
            for seg in segments
            if seg.strip()
        ]
        results = [r for r in results if r.intent != "unknown"]
        return results if results else [self.classify(text)]

    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="unknown", confidence=0.0, entities={}, raw_text=text)

        # 1. High-priority keyword override (Stop/Emergency)
        text_clean = text.strip().lower()
        if any(w in text_clean for w in ["stop", "cancel", "shut up", "nevermind"]):
            return IntentResult(intent="stop", confidence=1.0, entities={}, raw_text=text)

        # 1b. Vehicle control keyword override (reliable, no ML needed)
        _vc_components = ["ac", "dac", "heater", "fan", "window", "wiper", "horn",
                          "lights", "light", "seat", "sunroof", "sun roof", "roof",
                          "sound roof", "defrost", "air conditioning", "air conditioner", "temperature",
                          "camera", "rear camera", "backup camera", "reverse camera"]
        _vc_action_words = ["on", "off", "open", "close", "up", "down",
                            "increase", "decrease", "higher", "lower",
                            "switch", "turn", "adjust", "set"]
        _has_component = any(c in text_clean for c in _vc_components)
        _has_action    = any(a in text_clean for a in _vc_action_words)
        _question_words = ["what", "how", "is", "are", "check", "status", "show", "tell"]
        _bare_component = (len(text_clean.split()) <= 2
                           and not any(q in text_clean for q in _question_words))
        if _has_component and (_has_action or _bare_component):
            entities = extract_entities(text, "vehicle_control")
            if entities.get("component"):
                return IntentResult(intent="vehicle_control", confidence=0.95,
                                    entities=entities, raw_text=text)

        # 1c. Navigation keyword override
        _nav_triggers = ["navigate", "go to", "take me to", "get me to", "drive to",
                         "route to", "directions to", "head to", "how do i get to"]
        if any(t in text_clean for t in _nav_triggers):
            entities = extract_entities(text, "navigation")
            return IntentResult(intent="navigation", confidence=0.95,
                                entities=entities, raw_text=text)

        # 1d. Implicit comfort/state phrases → vehicle control (no component keyword needed)
        _comfort_map = [
            (["i'm cold", "im cold", "i am cold", "it's cold", "its cold",
              "too cold", "feeling cold", "so cold", "freezing",
              "very cold", "bit cold", "quite cold", "pretty cold",
              "getting cold", "cold in here", "chilly", "i feel cold",
              "it feels cold", "so chilly", "really cold"],
             {"component": "heater", "action": "on"}),
            (["i'm hot", "im hot", "i am hot", "it's hot", "its hot",
              "too hot", "feeling hot", "so hot", "getting warm",
              "it's warm", "its warm", "too warm", "sweating",
              "very hot", "really hot", "burning up", "boiling",
              "roasting", "hot in here", "i feel hot", "it feels hot"],
             {"component": "ac", "action": "on"}),
            (["stuffy", "need fresh air", "hard to breathe", "need some air",
              "can't breathe", "no air"],
             {"component": "ac", "action": "on"}),
        ]
        for phrases, entities in _comfort_map:
            if any(p in text_clean for p in phrases):
                return IntentResult(intent="vehicle_control", confidence=0.88,
                                    entities=entities, raw_text=text)

        # 1e. Bare temperature value commands → AC climate control
        #     e.g. "increase to 15 degrees", "set to 22c", "make it 20 celsius"
        if re.search(r"\b\d+\s*(?:c\b|°c|degrees?(?:\s+celsius)?)\b", text_clean):
            return IntentResult(intent="vehicle_control", confidence=0.87, entities={}, raw_text=text)

        if not self.recognizer:
            # Fallback to general question if ML fails
            return IntentResult(intent="general_question", confidence=0.5, entities={}, raw_text=text)

        # 2. Semantic Embedding Classification
        self.latest_match = None
        self.recognizer.process_utterance(text_clean)
        
        if self.latest_match:
            best_intent = self.trigger_to_intent[self.latest_match.trigger_phrase]
            confidence = round(self.latest_match.similarity, 2)
        else:
            best_intent = "general_question"
            confidence = 0.5
            
        # 3. Knowledge override
        if best_intent != "general_question":
            knowledge_keywords = [r"\bnews\b", r"\bweather\b", r"\btell me about\b",
                                  r"\bwhat is\b", r"\bwho is\b", r"\bknow about\b",
                                  r"\blatest\b", r"\bwhat happened\b", r"\btoday\b"]
            if any(re.search(p, text_clean) for p in knowledge_keywords):
                best_intent = "general_question"
                confidence = 0.9

        entities = extract_entities(text, best_intent)

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities,
            raw_text=text
        )

if __name__ == "__main__":
    classifier = IntentClassifier()

    test_commands = [
        "turn on the AC",
        "navigate to the airport",
        "play some relaxing music",
        "call mom",
        "stop",
        "what is the weather today",
        "order a frappuccino from Starbucks",
        "I want a caramel latte",
        "get me a coffee",
        "order from Subway",
        "I'd like an espresso",
        "buy me a cold brew from Blue Bottle",
        "order food",
        "I want a veggie delight from Subway",
    ]

    print("\n" + "="*60)
    print("  NOVA Layer 7 — Intent Classifier Test (Moonshine ML)")
    print("="*60)

    for cmd in test_commands:
        result = classifier.classify(cmd)
        print(f"\n  Input     : {cmd}")
        print(f"  Intent    : {result.intent}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Entities  : {result.entities}")
        print("  " + "-"*55)
