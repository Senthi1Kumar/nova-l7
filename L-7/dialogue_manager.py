"""
NOVA Layer 7 - Dialogue Manager

AGREED FLOW:
──────────────────────────────────────────────────
1. Voice verification required for payment — NEEDS_VERIFICATION = ["payment"]
2. Payment flow: order → merchant → yes → voice verify → Voice OTP → payment
3. OTP fail x2 → PIN fallback → payment denied if PIN fails
4. Location check after OTP pass
5. Session expiry warning at < 2 min (session_start tracked)
6. Full L3 auth available via: python layer3_main.py (terminal only)
7. All other features: emergency, stop, pause, resume, help, buffer
──────────────────────────────────────────────────
"""

from intent_classifier import IntentClassifier, IntentResult
from dataclasses import dataclass, field
from typing import Optional
import time
import random
import sys
import os

# ── Connect to Layer 3 ────────────────────────────────────────────────────────
L3_PATH = os.path.join(os.path.dirname(__file__), '..', 'L-3')
sys.path.insert(0, os.path.abspath(L3_PATH))

L3_AVAILABLE = False
try:
    from layer3_main import check_session, authorize_payment, _create_and_register_token
    from verify import verify_voice
    from pin_handler import verify_pin
    from face_handler import verify_face
    L3_AVAILABLE = True
    print("[Layer 7] Layer 3 connected — session/PIN checking active")
except ImportError as e:
    print(f"[Layer 7] Layer 3 not available ({e}) — simulation mode")

# ── Connect to Mock Commerce ──────────────────────────────────────────────────
COMMERCE_AVAILABLE = False
try:
    from mock_order import (
        search_merchants, get_menu, create_basket,
        add_to_basket, checkout, process_payment
    )
    COMMERCE_AVAILABLE = True
    print("[Layer 7] Mock commerce connected — SQLite backend ready")
except ImportError as e:
    print(f"[Layer 7] Commerce not available ({e})")

# ── Connect to Audit Log ──────────────────────────────────────────────────────
AUDIT_AVAILABLE = False
try:
    from audit_log import log_event
    AUDIT_AVAILABLE = True
    print("[Layer 7] Audit log connected")
except ImportError:
    print("[Layer 7] Audit log not available — skipping logging")


# ── Constants ─────────────────────────────────────────────────────────────────
NEEDS_VERIFICATION     = ["payment"]  # voice verification required before payment OTP
CONTEXT_DECAY_SECONDS  = 300
QUEUE_MAX              = 5
OTP_MAX_ATTEMPTS       = 2
PIN_MAX_ATTEMPTS       = 3
SESSION_DURATION       = 900         # 15 minutes
SESSION_WARN_SECONDS   = 120         # warn when < 2 min remaining


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class ConversationTurn:
    role: str
    text: str
    intent: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueuedCommand:
    text: str
    received_at: float = field(default_factory=time.time)


@dataclass
class DialogueState:
    fsm_state: str = "IDLE"
    current_intent: str = None
    pending_entities: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    slot_attempt: int = 0
    confirm_action: str = None
    confirm_details: dict = field(default_factory=dict)

    # Session
    session_token: str = None
    session_valid: bool = False
    session_start: float = None
    session_duration: int = SESSION_DURATION

    # OTP / PIN tracking
    otp_attempts: int = 0
    pin_attempts: int = 0

    # Lockout
    locked_out: bool = False

    # Conversation
    history: list = field(default_factory=list)
    last_interaction: float = field(default_factory=time.time)
    last_vehicle_components: list = field(default_factory=list)

    # FIFO buffer
    command_buffer: list = field(default_factory=list)
    is_speaking: bool = False
    greeted: bool = False



# ── Profile path ──────────────────────────────────────────────────────────────
import json
import os
PROFILES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'L-3', 'data', 'profiles.json')
)

def load_driver_profile(driver_id: str) -> dict:
    try:
        if os.path.exists(PROFILES_PATH):
            with open(PROFILES_PATH, "r") as f:
                profiles = json.load(f)
            if driver_id in profiles:
                print(f"[Layer 7] Profile loaded – name: {profiles[driver_id].get('name')}")
                return profiles[driver_id]
    except Exception as e:
        print(f"[Layer 7] Could not load profile: {e}")
    return None

def save_profile_field(driver_id: str, key: str, value) -> None:
    try:
        profiles = {}
        if os.path.exists(PROFILES_PATH):
            with open(PROFILES_PATH, "r") as f:
                profiles = json.load(f)
        if driver_id not in profiles:
            profiles[driver_id] = {}
        profiles[driver_id][key] = value
        os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
        with open(PROFILES_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
    except Exception as e:
        print(f"[Layer 7] Profile save error: {e}")

def get_greeting(profile, driver_id: str = "driver1") -> str:
    if profile is None:
        return "Hey Guest! I'm Nova, your in-car assistant. How can I help you today?"
    name   = (profile.get("name") or "").strip()
    orders = profile.get("total_orders", 0)
    if not name or name.lower() in ("there", "guest", ""):
        label = driver_id.replace("driver", "Driver ").strip()
        return f"Hey {label}! I'm Nova, your in-car assistant. How can I help you today?"
    last_order = profile.get("last_order")
    if orders > 0 and last_order:
        merchant = last_order.get("merchant_name", "")
        item     = last_order.get("item_name", "")
        return f"Welcome back, {name}. Your usual {item} from {merchant} is ready to order. Say 'usual' to reorder, or let me know what you need."
    if orders > 0:
        return f"Welcome back, {name}! You have {orders} order{'s' if orders > 1 else ''} with Nova. How can I help?"
    return f"Hey {name}! I'm Nova, your in-car assistant. How can I help you today?"

# ── Response builder ──────────────────────────────────────────────────────────
def build_response(
    message: str,
    state: str,
    intent: str,
    routing: str,
    entities: dict = None,
    action: str = None,
    otp: list = None,
    verification_status: str = None,
    queued: bool = False,
    queue_position: int = None,
    buffered_response: dict = None,
    session_warning: str = None,
    **kwargs
) -> dict:
    response = {
        "nova_says":           message,
        "fsm_state":           state,
        "intent":              intent,
        "routing":             routing,
        "entities":            entities or {},
        "action":              action,
        "otp":                 otp,
        "verification_status": verification_status,
        "queued":              queued,
        "queue_position":      queue_position,
        "buffered_response":   buffered_response,
        "session_warning":     session_warning,
        "timestamp":           time.time()
    }
    response.update(kwargs)
    return response


# ── Required slots ────────────────────────────────────────────────────────────
REQUIRED_SLOTS = {
    "navigation":      ["destination"],
    "communication":   ["contact"],
    "payment":         [],
    "vehicle_control": ["component", "action"],
    "media":           [],
    "general_question":[],
    "stop":            [],
    "emergency":       [],
    "help":            [],
    "pause":           [],
    "resume":          [],
    "queue_status":    [],
}

SLOT_QUESTIONS = {
    "destination": "Where would you like to go?",
    "contact":     "Who would you like to call or message?",
    "component":   "Which component — AC, window, heater, sunroof, or lights?",
    "action":      "Should I turn it on or off?",
}

# ── Keyword groups ────────────────────────────────────────────────────────────
EMERGENCY_KEYWORDS = ["emergency", "accident", "help me", "call 911", "call ambulance"]
STOP_KEYWORDS      = ["stop", "cancel", "quit", "shut up", "nevermind", "never mind"]
REPEAT_KEYWORDS    = ["repeat", "say that again", "say again", "what did you say"]
PAUSE_KEYWORDS     = ["pause", "hold on", "wait", "one second", "one moment"]
RESUME_KEYWORDS    = ["resume", "continue", "go ahead", "carry on"]
HELP_KEYWORDS      = ["help", "what can you do", "what do you know", "commands"]
QUEUE_KEYWORDS     = ["what's in the queue", "what is in the queue",
                      "how many requests", "what are you processing", "queue status"]
USUAL_KEYWORDS     = ["usual", "same as before", "same order", "my usual", "the usual", "repeat my order", "order again"]
CONFIRM_YES        = ["yes", "yeah", "yep", "confirm", "proceed",
                      "ok", "okay", "sure", "do it", "correct"]
CONFIRM_NO         = ["no", "nope", "cancel", "stop", "don't", "abort"]

NUMBER_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9
}


# ── Route handlers ────────────────────────────────────────────────────────────
# TTS-friendly display names for vehicle components
_COMPONENT_DISPLAY = {
    "ac": "A C", "rear_camera": "rear camera", "rear camera": "rear camera",
}

def _tts_component(name: str) -> str:
    """Return a TTS-friendly display name for a vehicle component."""
    return _COMPONENT_DISPLAY.get(name, name.capitalize())

def handle_vehicle_control(entities: dict) -> dict:
    commands = entities.get("commands")
    if commands and len(commands) > 1:
        # Compound command: "switch off the AC and open the sunroof"
        parts = [f"{_tts_component(c['component'])} turned {c['action']}" for c in commands]
        message = "Done. " + ", ".join(parts) + "."
    else:
        component = entities.get("component", "system")
        action    = entities.get("action", "on")
        message = f"Done. {_tts_component(component)} turned {action}."
    if entities.get("temperature_c"):
        message += f" Temperature set to {entities['temperature_c']}°C."
    if entities.get("fan_speed"):
        message += f" Fan speed: {entities['fan_speed']}."
    return build_response(
        message=message,
        state="IDLE", intent="vehicle_control",
        routing="Direct hardware command via CAN bus — no AI, no cost, under 50ms",
        entities=entities, action="hardware_command"
    )

def handle_navigation(entities: dict) -> dict:
    destination = entities.get("destination", "your destination")
    eta_minutes = random.randint(10, 45)
    distance_km = round(random.uniform(5, 30), 1)
    return build_response(
        message=f"Navigating to {destination}. Estimated {eta_minutes} minutes — {distance_km} km via fastest route.",
        state="IDLE", intent="navigation",
        routing="Maps API call — HERE / Google Maps — no LLM needed",
        entities=entities, action="maps_api_call"
    )

def handle_media(entities: dict) -> dict:
    query = entities.get("query", "your music")
    return build_response(
        message=f"Playing {query} on Spotify.",
        state="IDLE", intent="media",
        routing="Spotify API — direct call — no AI, no cost",
        entities=entities, action="spotify_api_call"
    )

def handle_general_question(text: str, history: list) -> dict:
    return build_response(
        message="Let me think about that...",
        state="IDLE", intent="general_question",
        routing="Local LLM with context",
        entities={}, action="llm_call",
        original_text=text
    )

def handle_communication(entities: dict) -> dict:
    contact = entities.get("contact", "contact")
    return build_response(
        message=f"Calling {contact} now.",
        state="IDLE", intent="communication",
        routing="Phone API — direct call",
        entities=entities, action="phone_call"
    )


def handle_order_flow(entities: dict, raw_text: str, dm_state) -> dict:
    """
    Full order flow:
      1. Fresh search → multiple results → ask to pick
      2. Merchant pick by name or number → basket + checkout
      3. CONFIRM_PENDING → yes → OTP → location check → payment
    """

    # ── Merchant already listed — user is picking ─────────────────────────────
    if dm_state.pending_entities.get("merchants"):
        merchants  = dm_state.pending_entities["merchants"]
        text_lower = raw_text.lower()
        import re as _re

        # Ordinals checked before cardinals to avoid "one" matching inside "second one"
        number_map = [
            ("first",  0), ("second", 1), ("third",  2),
            (r"\b1\b", 0), (r"\b2\b", 1), (r"\b3\b", 2),
            (r"\bone\b", 0), (r"\btwo\b", 1), (r"\bthree\b", 2),
        ]
        chosen = None
        for pattern, idx in number_map:
            if _re.search(pattern, text_lower):
                if idx < len(merchants):
                    chosen = merchants[idx]
                    break

        if not chosen:
            for m in merchants:
                if m["name"].lower() in text_lower:
                    chosen = m
                    break

        if not chosen:
            chosen = merchants[0]

        basket        = create_basket(chosen["id"])
        menu_r        = get_menu(chosen["id"])
        checkout_data = {}
        first         = None

        if menu_r["found"] and menu_r["menu"]:
            first = menu_r["menu"][0]
            quantity = dm_state.pending_entities.get("quantity", 1)
            add_to_basket(basket["basket_id"], first["id"], quantity)
            checkout_data = checkout(basket["basket_id"])

        item_price = first["price"] if first else 0.0
        item_name  = first["name"]  if first else "order"
        total      = checkout_data.get("total",    item_price)
        nova_fee   = checkout_data.get("nova_fee", 0)

        dm_state.pending_entities.update({
            "merchant_id":   chosen["id"],
            "merchant_name": chosen["name"],
            "basket_id":     basket["basket_id"],
            "address":       chosen.get("address", ""),
            "eta_order":     chosen.get("eta_order", "10 min"),
            "item_name":     item_name,
            "item_price":    item_price,
            "lat":           chosen.get("lat"),
            "lng":           chosen.get("lng"),
            "items":         checkout_data.get("items", item_name),
            "subtotal":      checkout_data.get("subtotal", item_price),
            "nova_fee":      nova_fee,
            "total":         total,
            "merchants":     None,
        })

        dm_state.fsm_state = "CONFIRM_PENDING"
        msg = (
            f"{chosen['name']} — {item_name} ${item_price:.2f}. "
            f"Total ${total:.2f} including ${nova_fee:.2f} Nova fee. "
            f"Say YES to confirm."
        )
        return build_response(
            message=msg, state="CONFIRM_PENDING",
            intent="payment", routing="Merchant selected — basket ready",
            entities=dm_state.pending_entities, action="show_merchant_on_map"
        )

    # ── Fresh search ──────────────────────────────────────────────────────────
    query = (entities.get("merchant") or entities.get("item") or
             entities.get("query") or raw_text)

    # Stash remaining items from compound orders ("2 coffees and 2 pizzas")
    multi_items = entities.get("multi_items", [])
    if len(multi_items) > 1 and not dm_state.pending_entities.get("pending_order_queue"):
        dm_state.pending_entities["pending_order_queue"] = multi_items[1:]

    if COMMERCE_AVAILABLE:
        result = search_merchants(query)

        if result["found"]:
            merchants = result["merchants"]

            if len(merchants) == 1:
                chosen        = merchants[0]
                basket        = create_basket(chosen["id"])
                menu_r        = get_menu(chosen["id"])
                checkout_data = {}
                first         = None

                if menu_r["found"] and menu_r["menu"]:
                    first = menu_r["menu"][0]
                    quantity = dm_state.pending_entities.get("quantity", 1)
                    add_to_basket(basket["basket_id"], first["id"], quantity)
                    checkout_data = checkout(basket["basket_id"])

                item_price = first["price"] if first else 0.0
                item_name  = first["name"]  if first else "order"

                dm_state.pending_entities.update({
                    "merchant_id":   chosen["id"],
                    "merchant_name": chosen["name"],
                    "basket_id":     basket["basket_id"],
                    "address":       chosen.get("address", ""),
                    "eta_order":     chosen.get("eta_order", "10 min"),
                    "item_name":     item_name,
                    "item_price":    item_price,
                    "lat":           chosen.get("lat"),
                    "lng":           chosen.get("lng"),
                    "items":         checkout_data.get("items", item_name),
                    "subtotal":      checkout_data.get("subtotal", item_price),
                    "nova_fee":      checkout_data.get("nova_fee", 0),
                    "total":         checkout_data.get("total", item_price),
                })

                dm_state.fsm_state = "CONFIRM_PENDING"
                return build_response(
                    message=result["nova_says"], state="CONFIRM_PENDING",
                    intent="payment", routing="Single merchant — basket ready",
                    entities=dm_state.pending_entities, action="show_merchant_on_map"
                )

            else:
                dm_state.pending_entities["merchants"] = merchants
                dm_state.fsm_state = "CONFIRM_PENDING"
                return build_response(
                    message=result["nova_says"], state="CONFIRM_PENDING",
                    intent="payment", routing="Multiple merchants — waiting for selection",
                    entities=dm_state.pending_entities
                )

        return build_response(
            message=f"Could not find {query} nearby.",
            state="IDLE", intent="payment", routing="No results"
        )

    # Fallback
    return build_response(
        message="Found Starbucks 0.4 miles away. Frappuccino $6.50. Confirm?",
        state="CONFIRM_PENDING", intent="payment", routing="Fallback",
        entities={"merchant_name": "Starbucks", "item_name": "Frappuccino", "item_price": 6.50}
    )


def generate_otp() -> list:
    return [random.randint(0, 9) for _ in range(4)]

def otp_to_words(otp: list) -> str:
    words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    return ", ".join(words[d] for d in otp)

def _log_command(driver_id: str, command: str, intent: str, is_verified: bool):
    if not AUDIT_AVAILABLE:
        return
    try:
        log_event(
            event_type="COMMAND", driver_id=driver_id, passed=is_verified,
            details={"command": command, "intent": intent, "is_verified": is_verified}
        )
    except Exception as e:
        print(f"[audit] Log failed: {e}")


# ── Main Dialogue Manager ─────────────────────────────────────────────────────
class DialogueManager:

    def __init__(self):
        self.classifier          = IntentClassifier()
        self.state               = DialogueState()
        self._otp_active         = None
        self._driver_id          = "driver1"
        self._paused_intent      = None
        self._paused_entities    = None
        self._last_nova_response = None

        self._profile     = load_driver_profile(self._driver_id)
        self._driver_name = (self._profile or {}).get("name", "there") or "there"

        if self._profile:
            save_profile_field(self._driver_id, "last_seen", time.strftime("%Y-%m-%d %H:%M"))
            print(f"[Layer 7] Driver: {self._driver_name}")
        else:
            self._driver_name = "Guest"


    # ══════════════════════════════════════════════════════════════════════════
    # BUFFER / QUEUE
    # ══════════════════════════════════════════════════════════════════════════

    
    def _get_greeting_response(self) -> dict:
        greeting = get_greeting(self._profile, self._driver_id)
        return self._store_response(build_response(
            message=greeting,
            state="IDLE", intent="greeting",
            routing="Profile loaded – personalized greeting",
            driver_name=self._driver_name
        ))

    def speaking_started(self):
        self.state.is_speaking = True
        print("[dialogue] Nova started speaking — buffer active")

    def speaking_done(self) -> Optional[dict]:
        self.state.is_speaking = False
        print("[dialogue] Nova finished speaking — checking buffer...")
        if self.state.command_buffer:
            next_cmd  = self.state.command_buffer.pop(0)
            wait_time = round(time.time() - next_cmd.received_at, 1)
            response  = self._process_command(next_cmd.text)
            response["from_buffer"]   = True
            response["buffer_waited"] = wait_time
            if self.state.command_buffer:
                response["nova_says"] += f" ({len(self.state.command_buffer)} more in queue)"
            return response
        print("[dialogue] Buffer empty — returning to IDLE")
        return None

    def _push_to_buffer(self, text: str) -> dict:
        position = len(self.state.command_buffer) + 1
        self.state.command_buffer.append(QueuedCommand(text=text))
        print(f"[dialogue] Buffered: '{text}' at position {position}")
        return build_response(
            message="Got it. I'll answer that next — finishing current response first.",
            state="BUFFERED", intent="buffered",
            routing=f"Command buffered at position {position}",
            queued=True, queue_position=position
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION
    # ══════════════════════════════════════════════════════════════════════════

    def _is_session_valid(self) -> bool:
        if not self.state.session_valid or not self.state.session_token:
            return False
        if L3_AVAILABLE:
            try:
                return check_session(self.state.session_token).get("valid", False)
            except Exception:
                return False
        return (time.time() - (self.state.session_start or 0)) < self.state.session_duration

    def _run_layer3_auth(self, audio_buffer=None) -> tuple:
        if L3_AVAILABLE:
            print("[Layer 7] Calling Layer 3 authenticate()...")
            token = authenticate(self._driver_id, audio_buffer=audio_buffer)
            return (True, token) if token else (False, "Authentication failed")
        print("[Layer 7] Simulation mode — auto-passing verification")
        self.state.session_start = time.time()
        self.state.session_valid = True
        return True, "simulated_token"

    def _session_time_remaining(self) -> int:
        """Returns seconds remaining in session. 0 if expired."""
        if not self.state.session_start:
            return 0
        elapsed = time.time() - self.state.session_start
        remaining = int(self.state.session_duration - elapsed)
        return max(0, remaining)

    def _get_session_warning(self) -> Optional[str]:
        """Returns warning string if session expiring soon, else None."""
        remaining = self._session_time_remaining()
        if 0 < remaining <= SESSION_WARN_SECONDS:
            mins = remaining // 60
            secs = remaining % 60
            if mins > 0:
                return f"Session expires in {mins} min {secs} sec."
            return f"Session expires in {secs} seconds."
        return None

    def _start_session(self, token: str):
        self.state.session_token = token
        self.state.session_valid = True
        self.state.session_start = time.time()
        self.state.locked_out    = False

    def _check_context_decay(self):
        if (time.time() - self.state.last_interaction) > CONTEXT_DECAY_SECONDS:
            self.state.history = []

    def _add_to_history(self, role: str, text: str, intent: str):
        self.state.history.append(ConversationTurn(role=role, text=text, intent=intent))
        if len(self.state.history) > 10:
            self.state.history = self.state.history[-10:]
        self.state.last_interaction = time.time()

    def _resolve_context(self, text: str) -> str:
        import re
        text_lower        = text.lower()
        last_turns        = self.state.history[-2:]
        last_destinations = []
        last_contacts     = []
        for turn in last_turns:
            if turn.intent == "navigation":
                m = re.search(r"(?:to|at|near)\s+([A-Z][a-zA-Z\s]+)", turn.text)
                if m:
                    last_destinations.append(m.group(1).strip())
            if turn.intent == "communication":
                m = re.search(r"call\s+(\w+)", turn.text.lower())
                if m:
                    last_contacts.append(m.group(1))
        if any(w in text_lower for w in ["there", "that place", "same place"]) and last_destinations:
            text = re.sub(r"\b(there|that place|same place)\b", last_destinations[-1],
                          text, flags=re.IGNORECASE)
        if any(w in text_lower for w in ["him", "her", "them"]) and last_contacts:
            text = re.sub(r"\b(him|her|them)\b", last_contacts[-1],
                          text, flags=re.IGNORECASE)
        # Vehicle control context: resolve "both"/"them"/"those" → last acted components
        if self.state.last_vehicle_components:
            comps = self.state.last_vehicle_components
            comp_str = " and ".join(comps)
            if any(w in text_lower for w in ["both", "those", "all of them", "all of those"]):
                text = re.sub(r"\b(both|those|all of them|all of those)\b",
                              comp_str, text, flags=re.IGNORECASE)
            elif "them" in text_lower and not last_contacts:
                text = re.sub(r"\bthem\b", comp_str, text, flags=re.IGNORECASE)
            elif len(comps) == 1 and any(w in text_lower for w in ["it", "that"]):
                # Don't expand "it"/"that" when the clause is a value-setting command
                # e.g. "set it to 10c" / "put it on 20 degrees" — "it" refers to a value, not component
                import re as _re
                is_value_setting = bool(_re.search(r"set\s+\w+\s+to\s+\d|to\s+\d+\s*(?:c\b|degrees?)", text_lower))
                if not is_value_setting:
                    text = re.sub(r"\b(it|that)\b", comps[0], text, flags=re.IGNORECASE)
        return text

    def _check_missing_slots(self, intent: str, entities: dict) -> list:
        return [s for s in REQUIRED_SLOTS.get(intent, []) if not entities.get(s)]

    def _store_response(self, response: dict) -> dict:
        if response.get("nova_says"):
            self._last_nova_response = response["nova_says"]
            self._add_to_history("nova", response["nova_says"], response.get("intent", ""))
        # Attach session warning to every response if applicable
        warning = self._get_session_warning()
        if warning and not response.get("session_warning"):
            response["session_warning"] = warning
        response["driver_name"] = self._driver_name
        return response

    def _check_location(self, entities: dict) -> dict:
        """Known GPS (lat/lng from merchant) → auto-confirm. Unknown → ask driver."""
        if entities.get("lat") and entities.get("lng"):
            return {"known": True, "confirmed": True}
        return {"known": False, "confirmed": False}

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def process(self, user_input: str, audio_buffer=None) -> dict:
        """
        Main entry point — call this for every user message.
        """
        user_input = user_input.strip()
        if not user_input:
            return build_response(message="I didn't catch that. Could you repeat?",
                                  state="IDLE", intent="unknown", routing="None")

        if not self.state.greeted:
            self.state.greeted = True
            self._add_to_history("user", user_input, "greeting")
            return self._get_greeting_response()

        # Lockout — absolute block
        if self.state.locked_out:
            return build_response(
                message="System is locked. All authentication attempts failed. Please contact support.",
                state="LOCKED", intent="lockout",
                routing="Full lockout — all 3 auth levels failed",
                verification_status="locked"
            )

        text_lower   = user_input.lower().strip()
        is_interrupt = (any(kw in text_lower for kw in EMERGENCY_KEYWORDS) or
                        any(kw in text_lower for kw in STOP_KEYWORDS))

        if self.state.is_speaking and not is_interrupt:
            if len(self.state.command_buffer) >= QUEUE_MAX:
                return build_response(
                    message="I already have several questions waiting. Please wait a moment.",
                    state="BUFFERED", intent="queue_full",
                    routing=f"Buffer full — max {QUEUE_MAX} items"
                )
            return self._push_to_buffer(user_input)

        # ── Not speaking or is interrupt — process normally ───────────────────
        return self._process_command(user_input, audio_buffer=audio_buffer)

    def _process_command(self, user_input: str, audio_buffer=None) -> dict:
        """
        Internal — actually processes the command.
        Called directly when not speaking, or from speaking_done() for buffered commands.
        """
        self._check_context_decay()
        text_lower = user_input.lower().strip()

        # ── Emergency ─────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            self.state.is_speaking    = False
            self.state.command_buffer.clear()
            self.state.fsm_state      = "IDLE"
            self.state.current_intent = None
            self._otp_active          = None
            _log_command(self._driver_id, user_input, "emergency", self._is_session_valid())
            return self._store_response(build_response(
                message="Emergency detected. Automatically calling 911 and alerting your family members, Mom and Dad. Please stay calm.",
                state="IDLE", intent="emergency",
                routing="Emergency — absolute highest priority — buffer cleared",
                action="emergency_call"
            ))

        # ── Stop ──────────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in STOP_KEYWORDS):
            self.state.is_speaking    = False
            self.state.command_buffer.clear()
            self.state.fsm_state      = "IDLE"
            self.state.current_intent = None
            self._otp_active          = None
            self._paused_intent       = None
            self.state.otp_attempts   = 0
            self.state.pin_attempts   = 0
            _log_command(self._driver_id, user_input, "stop", self._is_session_valid())
            return self._store_response(build_response(
                message="Stopped. Buffer and queue cleared. Ready for commands.",
                state="IDLE", intent="stop",
                routing="Interrupt — immediate halt — buffer + queue flushed"
            ))

        # ── Greeting (explicit "hi"/"hey"/"hello") ─────────────────────────
        _greeting_words = {"hey", "hello", "hi", "good morning", "good evening",
                           "good afternoon", "hey there", "hi there", "howdy", "hey nova",
                           "hi nova", "hello nova"}
        # Only match when the *entire* utterance is a greeting — no startswith()
        # because "Hey turn on the AC." starts with "hey " and would be swallowed.
        if text_lower.strip().rstrip(".,!?") in _greeting_words:
            greeting = get_greeting(self._profile, self._driver_id)
            self._add_to_history("user", user_input, "greeting")
            return self._store_response(build_response(
                message=greeting,
                state="IDLE", intent="greeting",
                routing="Greeting detected — profile-based response (no LLM)"
            ))

        # ── Repeat ────────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in REPEAT_KEYWORDS):
            msg = self._last_nova_response or "Nothing to repeat yet."
            return self._store_response(build_response(
                message=msg, state="IDLE", intent="repeat", routing="Repeat last response"
            ))

        # ── Pause ─────────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in PAUSE_KEYWORDS):
            if self.state.current_intent:
                self._paused_intent   = self.state.current_intent
                self._paused_entities = self.state.pending_entities.copy()
                self.state.fsm_state  = "IDLE"
                return self._store_response(build_response(
                    message=f"Paused. Saved your {self._paused_intent} request. Say resume when ready.",
                    state="IDLE", intent="pause", routing="Paused — intent saved"
                ))
            return self._store_response(build_response(
                message="Nothing active to pause.", state="IDLE",
                intent="pause", routing="Nothing to pause"
            ))

        # ── Resume ────────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in RESUME_KEYWORDS):
            if self._paused_intent:
                restored_intent       = self._paused_intent
                restored_entities     = self._paused_entities
                self._paused_intent   = None
                self._paused_entities = None
                self.state.current_intent   = restored_intent
                self.state.pending_entities = restored_entities
                filled   = IntentResult(intent=restored_intent, confidence=1.0,
                                        entities=restored_entities, raw_text=user_input)
                response = self._route(filled)
                response["resume_message"] = f"Resuming your {restored_intent} request."
                return self._store_response(response)
            return self._store_response(build_response(
                message="Nothing to resume.", state="IDLE",
                intent="resume", routing="No paused context"
            ))

        # ── Help ──────────────────────────────────────────────────────────────
        if any(kw in text_lower for kw in HELP_KEYWORDS):
            return self._store_response(build_response(
                message=(
                    "Here's what I can help with: "
                    "Vehicle controls — AC, windows, lights. "
                    "Navigation — directions and routes. "
                    "Music — play, pause, skip. "
                    "Calls and messages — hands free. "
                    "Payments — order food, fuel, parking. "
                    "General questions — weather, news, anything. "
                    "Say stop to cancel. Say pause to hold. Say repeat to hear again."
                ),
                state="IDLE", intent="help", routing="Help message delivered"
            ))

        # ── Queue status ──────────────────────────────────────────────────────
        if any(kw in text_lower for kw in QUEUE_KEYWORDS):
            buf = self.state.command_buffer
            msg = ("Buffer is empty. Ready for your next command." if not buf else
                   f"I have {len(buf)} command{'s' if len(buf)>1 else ''} waiting: " +
                   ", ".join(f'"{c.text}"' for c in buf[:5]))
            return self._store_response(build_response(
                message=msg, state="IDLE", intent="queue_status",
                routing="Buffer status read aloud"
            ))

        # ── FSM states ────────────────────────────────────────────────────────
        if self.state.fsm_state == "SLOT_FILL":
            if self._should_escape_slot_fill(user_input):
                # User switched topic — reset and fall through to normal routing
                self.state.fsm_state = "IDLE"
                self.state.current_intent = None
                self.state.missing_slots = []
                self.state.pending_entities = {}
                self.state.slot_attempt = 0
            else:
                return self._store_response(self._handle_slot_fill(user_input))

        if self.state.fsm_state == "CONFIRM_PENDING":
            return self._store_response(self._handle_confirmation(user_input))

        if self.state.fsm_state == "OTP_PENDING":
            return self._store_response(self._handle_otp(user_input))

        if self.state.fsm_state == "VERIFY":
            return self._store_response(self._handle_verify_voice(user_input, audio_buffer))

        if self.state.fsm_state == "VERIFY_PIN":
            return self._store_response(self._handle_verify_pin(user_input))

        if self.state.fsm_state == "VERIFY_FACE":
            return self._store_response(self._handle_verify_face(user_input))

        if self.state.fsm_state == "PIN_PAYMENT_PENDING":
            return self._store_response(self._handle_pin_for_payment(user_input))

        if self.state.fsm_state == "LOCATION_CONFIRM":
            return self._store_response(self._handle_location_confirm(user_input))

        # ── Normal flow ───────────────────────────────────────────────────────
        
        if any(kw in text_lower for kw in USUAL_KEYWORDS):
            last = (self._profile or {}).get("last_order")
            if last and last.get("merchant_name") and last.get("item_name"):
                merchant_name = last["merchant_name"]
                item_name     = last["item_name"]
                item_price    = last.get("item_price", 0.0)
                nova_fee      = round(item_price * 0.03, 2)
                total         = round(item_price + nova_fee, 2)
                self.state.current_intent = "payment"
                self.state.pending_entities = {
                    "merchant_id":   last.get("merchant_id"),
                    "merchant_name": merchant_name,
                    "item_id":       last.get("item_id"),
                    "item_name":     item_name,
                    "item_price":    item_price,
                    "total":         total,
                    "nova_fee":      nova_fee,
                    "subtotal":      item_price,
                    "eta_order":     "8 min",
                }
                self.state.fsm_state = "CONFIRM_PENDING"
                return self._store_response(build_response(
                    message=(
                        f"Your usual order: {item_name} from {merchant_name}, "
                        f"${item_price:.2f}. Total ${total:.2f} including Nova fee. "
                        f"Say YES to confirm."
                    ),
                    state="CONFIRM_PENDING", intent="payment",
                    routing="Usual order shortcut – skipping merchant search",
                    entities=self.state.pending_entities
                ))
            else:
                return self._store_response(build_response(
                    message="You don't have a usual order yet! What would you like to order?",
                    state="IDLE", intent="payment",
                    routing="No usual order found"
                ))
        resolved_text = self._resolve_context(user_input)
        result        = self.classifier.classify(resolved_text)
        self._add_to_history("user", user_input, result.intent)

        _log_command(
            driver_id   = self._driver_id if self._is_session_valid() else "unknown_user",
            command     = user_input,
            intent      = result.intent,
            is_verified = self._is_session_valid()
        )

        # Session check for payment — warn if expiring soon
        if result.intent == "payment" and self._is_session_valid():
            warning = self._get_session_warning()
            if warning:
                print(f"[dialogue] Session warning: {warning}")

        # Default action for vehicle_control — if component is clear but action
        # wasn't extracted (e.g. STT dropped "open"/"close"), default to "on"
        # rather than asking "Should I turn it on or off?"
        if (result.intent == "vehicle_control"
                and result.entities.get("component")
                and not result.entities.get("action")):
            result.entities["action"] = "on"

        missing = self._check_missing_slots(result.intent, result.entities)
        if missing:
            self.state.fsm_state        = "SLOT_FILL"
            self.state.current_intent   = result.intent
            self.state.pending_entities = result.entities
            self.state.missing_slots    = missing
            self.state.slot_attempt     = 0
            question = SLOT_QUESTIONS.get(missing[0], f"What is the {missing[0]}?")
            return self._store_response(build_response(
                message=question, state="SLOT_FILL", intent=result.intent,
                routing=f"Slot filling — missing: {missing}",
                entities=result.entities
            ))

        return self._store_response(self._route(result))

    # ── Route ─────────────────────────────────────────────────────────────────
    def _route(self, result: IntentResult) -> dict:
        if result.intent == "vehicle_control":
            # Track which components were just acted on for context resolution
            commands = result.entities.get("commands")
            if commands and len(commands) > 1:
                self.state.last_vehicle_components = [c["component"] for c in commands if c.get("component")]
            elif result.entities.get("component"):
                self.state.last_vehicle_components = [result.entities["component"]]
            return handle_vehicle_control(result.entities)
        elif result.intent == "navigation":
            return handle_navigation(result.entities)
        elif result.intent == "media":
            return handle_media(result.entities)
        elif result.intent == "communication":
            return handle_communication(result.entities)
        elif result.intent == "general_question":
            return handle_general_question(result.raw_text, self.state.history)
        elif result.intent == "payment":
            self.state.current_intent   = "payment"
            self.state.pending_entities = result.entities
            return handle_order_flow(result.entities, result.raw_text, self.state)
        else:
            return handle_general_question(result.raw_text, self.state.history)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _start_otp(self, prefix_msg: str = "") -> dict:
        """Generate OTP and transition to OTP_PENDING. Used after verification passes."""
        self.state.fsm_state    = "OTP_PENDING"
        self.state.otp_attempts = 0
        otp = generate_otp()
        self._otp_active = otp
        msg = f"{prefix_msg} Voice OTP: Please repeat — {otp_to_words(otp)}.".strip()
        return build_response(
            message=msg,
            state="OTP_PENDING", intent="payment",
            routing="Voice OTP — unique per transaction",
            otp=otp,
            verification_status="passed",
            session_started=True,
        )

    def _handle_verify_voice(self, user_input: str, audio_buffer=None) -> dict:
        print(f"[Layer 7] Starting Layer 3 auth for: {self._driver_id}")
        
        if L3_AVAILABLE and audio_buffer is not None:
            # Try voice first
            result = verify_voice(self._driver_id, verbose=True, audio_buffer=audio_buffer)
            
            if result.get("error") or not result["passed"]:
                self.state.fsm_state = "VERIFY_PIN"
                return build_response(
                    message="Voice not recognized. Please say your PIN number.",
                    state="VERIFY_PIN", intent="verify",
                    routing="Layer 3 — voice failed, falling back to PIN",
                    verification_status="needed"
                )
            
            # Voice passed! Create token.
            token = _create_and_register_token(self._driver_id, "voice", verbose=True)
            self._start_session(token)

            # Payment flow: proceed to OTP instead of re-routing
            if self.state.current_intent == "payment":
                return self._start_otp("Voice recognized. Identity verified.")

            self.state.fsm_state = "IDLE"
            filled = IntentResult(
                intent=self.state.current_intent, confidence=1.0,
                entities=self.state.pending_entities, raw_text=user_input
            )
            response = self._route(filled)
            response["verify_message"]      = "Voice recognized. Identity verified."
            response["session_started"]     = True
            response["verification_status"] = "passed"
            return response

        elif not L3_AVAILABLE:
            # Simulation
            self.state.session_start = time.time()
            self.state.session_valid = True

            if self.state.current_intent == "payment":
                return self._start_otp("Identity verified. ")

            self.state.fsm_state = "IDLE"
            filled = IntentResult(
                intent=self.state.current_intent, confidence=1.0,
                entities=self.state.pending_entities, raw_text=user_input
            )
            response = self._route(filled)
            response["verify_message"]      = "Identity verified (simulation)."
            response["session_started"]     = True
            response["verification_status"] = "passed"
            return response
            
        return build_response(
            message="No audio buffer provided for verification.",
            state="IDLE", intent="verify",
            routing="Layer 3 — Missing audio buffer",
            action="verification_failed", verification_status="failed"
        )

    def _handle_verify_pin(self, user_input: str) -> dict:
        if L3_AVAILABLE:
            result = verify_pin(self._driver_id, user_input, verbose=True)
            if result.get("error") or not result["passed"]:
                self.state.fsm_state = "VERIFY_FACE"
                return build_response(
                    message="PIN incorrect. Please look at the cabin camera for Face ID.",
                    state="VERIFY_FACE", intent="verify",
                    routing="Layer 3 — PIN failed, falling back to Face ID",
                    verification_status="needed"
                )
                
            token = _create_and_register_token(self._driver_id, "pin", verbose=True)
            self._start_session(token)

            if self.state.current_intent == "payment":
                return self._start_otp("PIN accepted. Identity verified.")

            self.state.fsm_state = "IDLE"
            filled = IntentResult(
                intent=self.state.current_intent, confidence=1.0,
                entities=self.state.pending_entities, raw_text=user_input
            )
            response = self._route(filled)
            response["verify_message"]      = "PIN accepted. Identity verified."
            response["session_started"]     = True
            response["verification_status"] = "passed"
            return response

    def _handle_verify_face(self, user_input: str) -> dict:
        if L3_AVAILABLE:
            result = verify_face(self._driver_id, verbose=True)
            if result.get("error") or not result["passed"]:
                self.state.fsm_state = "IDLE"
                self.state.current_intent = None
                return build_response(
                    message="Face ID failed. Identity verification failed. Access denied.",
                    state="IDLE", intent="verify",
                    routing="Layer 3 — all levels failed",
                    action="verification_failed", verification_status="failed"
                )

            token = _create_and_register_token(self._driver_id, "face", verbose=True)
            self._start_session(token)

            if self.state.current_intent == "payment":
                return self._start_otp("Face recognized. Identity verified.")

            self.state.fsm_state = "IDLE"
            filled = IntentResult(
                intent=self.state.current_intent, confidence=1.0,
                entities=self.state.pending_entities, raw_text=user_input
            )
            response = self._route(filled)
            response["verify_message"]      = "Face recognized. Identity verified."
            response["session_started"]     = True
            response["verification_status"] = "passed"
            return response

    @staticmethod
    def _should_escape_slot_fill(user_input: str) -> bool:
        """Return True if the input looks like a new intent rather than a slot answer."""
        text = user_input.lower().strip()
        if any(w in text for w in ["cancel", "never mind", "nevermind", "forget it", "stop", "abort"]):
            return True
        if "?" in user_input:
            return True
        if len(user_input.split()) > 6:
            return True
        return False

    @staticmethod
    def _normalize_slot_value(slot: str, raw_input: str) -> str | None:
        """Extract a canonical slot value from the user's free-form response.

        Returns None if the input cannot be mapped to a valid value
        (caller should re-ask).
        """
        import re as _re
        text = raw_input.lower().strip()
        # Normalize spaced/dotted initials: "a. c." → "ac", "a.c." → "ac"
        text = _re.sub(r'\b([a-z])\.\s*([a-z])\.?\s*\b', r'\1\2', text)

        if slot == "action":
            if any(w in text for w in ["off", "close", "shut", "disable"]):
                return "off"
            if any(w in text for w in ["on", "open", "start", "enable", "yes", "switch", "turn"]):
                return "on"
            return None

        if slot == "component":
            controls = [
                "ac", "window", "sunroof", "heater", "fan",
                "lights", "light", "seat", "wiper", "horn",
                "defrost", "camera", "rear camera", "rear_camera",
            ]
            aliases = {
                "roof": "sunroof", "a/c": "ac", "air conditioning": "ac",
                "air conditioner": "ac", "air condition": "ac",
                "temperature": "ac", "sound roof": "sunroof",
                "sun roof": "sunroof", "dac": "ac",
                "backup camera": "rear_camera", "back camera": "rear_camera",
                "reverse camera": "rear_camera",
            }
            for alias, canonical in aliases.items():
                if alias in text:
                    return canonical
            for ctrl in controls:
                if ctrl in text:
                    return ctrl
            return None

        # destination, contact, etc. — keep the raw input
        return raw_input.strip()

    def _handle_slot_fill(self, user_input: str) -> dict:
        self.state.slot_attempt += 1
        if self.state.slot_attempt > 3:
            self.state.fsm_state      = "IDLE"
            self.state.current_intent = None
            return build_response(
                message="I'm sorry, I couldn't understand. Please try again.",
                state="IDLE", intent=self.state.current_intent or "unknown",
                routing="Slot fill failed after 3 attempts"
            )
        missing_slot = self.state.missing_slots[0]

        # Normalize the raw input into a canonical value for this slot
        value = self._normalize_slot_value(missing_slot, user_input)
        if value is None:
            # Could not parse — re-ask the same slot
            question = SLOT_QUESTIONS.get(missing_slot,
                                          f"What is the {missing_slot}?")
            return build_response(
                message=f"Sorry, I didn't catch that. {question}",
                state="SLOT_FILL", intent=self.state.current_intent,
                routing=f"Slot fill — could not parse '{user_input}' for {missing_slot}",
                entities=self.state.pending_entities
            )

        self.state.pending_entities[missing_slot] = value
        self.state.missing_slots.pop(0)
        if self.state.missing_slots:
            question = SLOT_QUESTIONS.get(self.state.missing_slots[0],
                                          f"What is the {self.state.missing_slots[0]}?")
            return build_response(
                message=question, state="SLOT_FILL",
                intent=self.state.current_intent,
                routing=f"Slot filling — still missing: {self.state.missing_slots}",
                entities=self.state.pending_entities
            )
        self.state.fsm_state = "IDLE"
        filled = IntentResult(intent=self.state.current_intent, confidence=1.0,
                              entities=self.state.pending_entities.copy(), raw_text=user_input)
        return self._route(filled)

    def _handle_confirmation(self, user_input: str) -> dict:
        """CONFIRM_PENDING: merchant selection OR yes/no for order."""
        user_lower = user_input.lower().strip()

        # Merchant list pending — this is a selection not yes/no
        if self.state.pending_entities.get("merchants"):
            return handle_order_flow({}, user_input, self.state)

        if any(w in user_lower for w in CONFIRM_YES):
            # Voice verification required before OTP if not already verified
            if "payment" in NEEDS_VERIFICATION and not self._is_session_valid():
                self.state.fsm_state = "VERIFY"
                return build_response(
                    message="Order confirmed. Please say anything to verify your identity.",
                    state="VERIFY", intent="payment",
                    routing="Layer 3 — voice verification required before payment",
                    verification_status="needed"
                )
            # Already verified (or verification not required) — proceed to OTP
            self.state.fsm_state    = "OTP_PENDING"
            self.state.otp_attempts = 0
            otp = generate_otp()
            self._otp_active = otp
            return build_response(
                message=f"Voice OTP: Please repeat — {otp_to_words(otp)}.",
                state="OTP_PENDING", intent="payment",
                routing="Voice OTP — unique per transaction",
                otp=otp
            )
        elif any(w in user_lower for w in CONFIRM_NO):
            self.state.fsm_state      = "IDLE"
            self.state.confirm_action = None
            return build_response(
                message="Payment cancelled.",
                state="IDLE", intent="payment",
                routing="Payment cancelled by driver"
            )
        else:
            return build_response(
                message="Please say YES to confirm or NO to cancel.",
                state="CONFIRM_PENDING", intent="payment",
                routing="Waiting for YES or NO"
            )

    def _handle_otp(self, user_input: str) -> dict:
        """
        OTP match → location check → payment.
        Fail x OTP_MAX_ATTEMPTS → PIN fallback.
        """
        # Parse spoken digits — handles both "seven two eight one" and "7281"
        spoken = []
        for w in user_input.lower().split():
            w = w.strip(".,!?")
            if w in NUMBER_WORDS:
                spoken.append(NUMBER_WORDS[w])
            elif w.isdigit() and len(w) > 1:
                # Concatenated digits like "7281" → [7, 2, 8, 1]
                spoken.extend(int(c) for c in w)

        if spoken == self._otp_active:
            self._otp_active        = None
            self.state.otp_attempts = 0
            location = self._check_location(self.state.pending_entities)

            if location["known"]:
                return self._process_payment_final()
            else:
                self.state.fsm_state = "LOCATION_CONFIRM"
                return build_response(
                    message="OTP verified. GPS location is unfamiliar. Can you confirm you're at a safe location? Say YES to proceed.",
                    state="LOCATION_CONFIRM", intent="payment",
                    routing="Location check — unknown GPS — driver confirmation needed"
                )
        else:
            self.state.otp_attempts += 1

            if self.state.otp_attempts >= OTP_MAX_ATTEMPTS:
                self._otp_active          = None
                self.state.otp_attempts   = 0
                self.state.pin_attempts   = 0
                self.state.fsm_state      = "PIN_PAYMENT_PENDING"
                return build_response(
                    message="OTP did not match. Please say your PIN to authorize payment instead.",
                    state="PIN_PAYMENT_PENDING", intent="payment",
                    routing="OTP failed max attempts — PIN fallback",
                    action="otp_failed_pin_fallback"
                )

            new_otp          = generate_otp()
            self._otp_active = new_otp
            return build_response(
                message=f"OTP did not match. New code: {otp_to_words(new_otp)}. Please repeat.",
                state="OTP_PENDING", intent="payment",
                routing=f"OTP retry {self.state.otp_attempts}/{OTP_MAX_ATTEMPTS}",
                otp=new_otp
            )

    def _handle_pin_for_payment(self, user_input: str) -> dict:
        """PIN fallback — max PIN_MAX_ATTEMPTS then payment denied."""
        self.state.pin_attempts += 1
        pin_input = user_input.strip()

        pin_ok = False
        if L3_AVAILABLE:
            try:
                pin_ok = verify_pin(self._driver_id, pin_input)
            except Exception as e:
                print(f"[Layer 7] PIN verify error: {e}")
                pin_ok = pin_input.isdigit() and len(pin_input) == 4
        else:
            pin_ok = pin_input.isdigit() and len(pin_input) == 4

        if pin_ok:
            self.state.fsm_state    = "IDLE"
            self.state.pin_attempts = 0
            return self._process_payment_final()

        if self.state.pin_attempts >= PIN_MAX_ATTEMPTS:
            self.state.fsm_state      = "IDLE"
            self.state.pin_attempts   = 0
            self.state.current_intent = None
            return build_response(
                message="PIN verification failed. Payment denied. Please try again later.",
                state="IDLE", intent="payment",
                routing="PIN fallback failed — payment denied",
                action="payment_denied", verification_status="pin_failed"
            )

        remaining = PIN_MAX_ATTEMPTS - self.state.pin_attempts
        return build_response(
            message=f"Incorrect PIN. {remaining} attempt{'s' if remaining > 1 else ''} remaining.",
            state="PIN_PAYMENT_PENDING", intent="payment",
            routing=f"PIN attempt {self.state.pin_attempts}/{PIN_MAX_ATTEMPTS}"
        )

    def _handle_location_confirm(self, user_input: str) -> dict:
        """Driver confirms unfamiliar location."""
        if any(w in user_input.lower() for w in CONFIRM_YES):
            return self._process_payment_final()
        self.state.fsm_state = "IDLE"
        return build_response(
            message="Payment cancelled for safety. Location could not be confirmed.",
            state="IDLE", intent="payment",
            routing="Location check failed — payment denied",
            action="payment_denied"
        )

    def _process_payment_final(self) -> dict:
        """Process payment after OTP + location cleared."""
        self.state.fsm_state = "IDLE"
        entities = self.state.pending_entities

        # Grab compound order queue before we clear pending_entities
        pending_order_queue = list(entities.get("pending_order_queue") or [])

        # Start a simulated session when payment completes (tracks expiry)
        if not self.state.session_start:
            self._start_session(f"web_session_{int(time.time())}")

        current_orders = (self._profile or {}).get("total_orders", 0)
        new_orders = current_orders + 1
        if self._profile is not None:
            self._profile["total_orders"] = new_orders
            save_profile_field(self._driver_id, "total_orders", new_orders)
            last_order = {
                "merchant_id":   entities.get("merchant_id"),
                "merchant_name": entities.get("merchant_name"),
                "item_id":       entities.get("item_id"),
                "item_name":     entities.get("item_name"),
                "item_price":    entities.get("item_price"),
            }
            self._profile["last_order"] = last_order
            save_profile_field(self._driver_id, "last_order", last_order)

        if COMMERCE_AVAILABLE and entities.get("basket_id"):
            result = process_payment(entities["basket_id"], entities)
            payment_msg = result["nova_says"]
            payment_entities = {
                "order_id":         result["order_id"],
                "transaction_id":   result["transaction_id"],
                "merchant_name":    result["merchant_name"],
                "merchant_address": result["merchant_address"],
                "items":            result["items"],
                "total":            result["total"],
                "nova_fee":         result["nova_fee"],
                "eta_order":        result["eta_order"],
                "payment_method":   result["payment_method"],
                "lat":              entities.get("lat"),
                "lng":              entities.get("lng")
            }
        else:
            payment_msg = "OTP verified. Order confirmed. $6.50 charged via Nova Pay. Ready in 8 minutes."
            payment_entities = {"transaction_id": "TXN-DEMO001", "nova_fee": 0.20}

        # If there are more items in a compound order, chain the next one
        if pending_order_queue:
            next_item = pending_order_queue[0]
            remaining  = pending_order_queue[1:]
            self.state.pending_entities = {}
            next_entities = {
                "item":     next_item["item"],
                "quantity": next_item.get("quantity", 1),
            }
            if remaining:
                next_entities["pending_order_queue"] = remaining
            next_response = handle_order_flow(next_entities, next_item["item"], self.state)
            combined_msg = f"{payment_msg} Now ordering your {next_item['item']}. {next_response['message']}"
            return build_response(
                message=combined_msg,
                state=next_response["state"], intent="payment",
                routing="Nova Pay — compound order chained",
                action=next_response.get("action", "payment_confirmed"),
                entities={**payment_entities, **self.state.pending_entities}
            )

        return build_response(
            message=payment_msg,
            state="IDLE", intent="payment",
            routing="Nova Pay — Stripe test mode — transaction complete",
            action="payment_confirmed",
            entities=payment_entities
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dm = DialogueManager()

    print("\n" + "="*60)
    print("  NOVA Layer 7 — Test")
    print(f"  L3:{L3_AVAILABLE}  Commerce:{COMMERCE_AVAILABLE}  Audit:{AUDIT_AVAILABLE}")
    print("="*60)

    steps = [
        ("order a coffee", "Search"),
        ("Starbucks",      "Pick merchant"),
        ("yes",            "Confirm"),
    ]

    for cmd, label in steps:
        r = dm.process(cmd)
        print(f"\n  [{label}] {cmd}")
        print(f"  Nova  : {r['nova_says']}")
        print(f"  State : {r['fsm_state']}")
        if r.get("otp"):
            otp_words = " ".join(
                ["zero","one","two","three","four","five",
                 "six","seven","eight","nine"][d] for d in r["otp"]
            )
            otp_r = dm.process(otp_words)
            print(f"\n  [OTP: {otp_words}]")
            print(f"  Nova  : {otp_r['nova_says']}")
            print(f"  State : {otp_r['fsm_state']}")
            e = otp_r.get("entities", {})
            if e.get("total"):
                print(f"  Total : ${e['total']:.2f}  Fee: ${e['nova_fee']:.2f}")
                print(f"  TXN   : {e.get('transaction_id', '—')}")