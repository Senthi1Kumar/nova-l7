"""
layer3_main.py
--------------
The main Layer 3 orchestrator. This is what Layer 7 calls.

This file ties everything together:
  - enroll.py       → register drivers
  - verify.py       → voice check
  - pin_handler.py  → PIN fallback
  - face_handler.py → face ID fallback
  - session_manager.py → session tokens
  - audit_log.py    → tamper-proof logging

Three things Layer 7 calls from this file:

  1. authenticate(driver_id)
     → Runs full auth flow (voice → PIN → face)
     → Returns session token if any level passes
     → Returns None + lockout if all fail

  2. check_session(token)
     → Is the 15-minute session still valid?
     → Returns True/False + driver_id

  3. authorize_payment(driver_id, token, amount, merchant)
     → Handles payment-specific auth
     → Voice OTP generation and verification
     → Returns authorized/denied

Run this file directly to test the full flow:
  python layer3_main.py --driver driver1
"""

import sys
import time
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from verify         import verify_voice, verify_for_payment
from pin_handler    import prompt_and_verify_pin, verify_pin
from face_handler   import verify_face
from session_manager import create_token, check_token, revoke_token, register_token
from audit_log      import log_event, print_log

MAX_VOICE_RETRIES = 3


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE OTP — Unique per transaction
# ══════════════════════════════════════════════════════════════════════════════

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}


def generate_otp() -> list:
    """Generate a random 4-digit OTP as a list of integers."""
    return [random.randint(0, 9) for _ in range(4)]


def otp_to_words(otp: list) -> str:
    """Convert OTP digits to spoken words. [7,2,9,4] → 'seven, two, nine, four'"""
    return ", ".join(NUMBER_WORDS[d] for d in otp)


def normalize_spoken_otp(spoken: str) -> list:
    """
    Convert spoken OTP back to digit list.
    "seven two nine four" → [7, 2, 9, 4]
    """
    word_to_num = {v: k for k, v in NUMBER_WORDS.items()}
    spoken = spoken.lower().strip()
    tokens = spoken.replace(",", " ").split()
    digits = []
    for token in tokens:
        if token in word_to_num:
            digits.append(word_to_num[token])
        elif token.isdigit() and len(token) == 1:
            digits.append(int(token))
    return digits


def run_voice_otp(driver_id: str, verbose: bool = True) -> dict:
    """
    Run the Voice OTP challenge for payment authorization.

    Flow:
      1. Generate random 4-digit OTP
      2. Nova speaks it aloud (simulated with print for demo)
      3. Driver repeats it back
      4. Check: correct voice + correct sequence

    Returns:
    {
        "passed":  True/False,
        "reason":  "OTP matched" or "Wrong sequence" or "Voice mismatch"
    }
    """
    otp = generate_otp()
    otp_words = otp_to_words(otp)

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  🔊 Nova says: \"Please repeat — {otp_words}\"")
        print(f"  {'─'*50}")

    # In real system: Whisper transcribes spoken response
    # For demo: typed input
    spoken = input(f"  You (type the words): ")

    spoken_digits = normalize_spoken_otp(spoken)

    if verbose:
        print(f"\n  [OTP] Expected  : {otp}")
        print(f"  [OTP] You said  : {spoken_digits}")

    if spoken_digits == otp:
        if verbose:
            print(f"  [OTP] ✅ Sequence matched!\n")
        log_event("OTP_PASS", driver_id, True, {"otp_length": 4})
        return {"passed": True, "reason": "OTP matched"}
    else:
        if verbose:
            print(f"  [OTP] ❌ Sequence mismatch.\n")
        log_event("OTP_FAIL", driver_id, False, {"expected": otp, "got": spoken_digits})
        return {"passed": False, "reason": "Wrong sequence"}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AUTH FLOW
# ══════════════════════════════════════════════════════════════════════════════

def authenticate(driver_id: str, verbose: bool = True, audio_buffer=None) -> Optional[str]:
    """
    Full 3-level authentication flow.
    """
    if verbose:
        print(f"\n{'═'*55}")
        print(f"  NOVA — IDENTITY VERIFICATION")
        print(f"  Driver: {driver_id}")
        print(f"{'═'*55}")

    # ── LEVEL 1: Voice ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  LEVEL 1 — Voice Biometric")

    for attempt in range(1, MAX_VOICE_RETRIES + 1):
        if verbose:
            print(f"\n  Attempt {attempt}/{MAX_VOICE_RETRIES}...")

        result = verify_voice(driver_id, verbose=verbose, audio_buffer=audio_buffer)

        if result.get("error"):
            if verbose:
                print(f"  ⚠️  {result['error']}")
            break

        log_event(
            "VOICE_PASS" if result["passed"] else "VOICE_FAIL",
            driver_id,
            result["passed"],
            {"score": result["score"], "threshold": result["threshold"],
             "attempt": attempt}
        )

        if result["passed"]:
            token = _create_and_register_token(driver_id, "voice", verbose)
            return token
            
        if audio_buffer is not None:
            # If using a pre-recorded buffer from the WebSocket, don't retry with the same buffer
            break

    if audio_buffer is not None:
        if verbose: print("  [fallback] Skipping interactive PIN/Face fallbacks because we are running in streaming mode.")
        return None

    if verbose:
        print(f"\n  ⚠️  Voice failed {MAX_VOICE_RETRIES} times. Moving to PIN fallback...")

    # ── LEVEL 2: PIN ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  LEVEL 2 — PIN Fallback")

    pin_result = prompt_and_verify_pin(driver_id, verbose=verbose)

    log_event(
        "PIN_PASS" if pin_result["passed"] else "PIN_FAIL",
        driver_id,
        pin_result["passed"],
        {"error": pin_result.get("error")}
    )

    if pin_result["passed"]:
        token = _create_and_register_token(driver_id, "pin", verbose)
        return token

    if verbose:
        print(f"\n  ⚠️  PIN failed. Moving to Face ID fallback...")

    # ── LEVEL 3: Face ID ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n  LEVEL 3 — Face ID Fallback")

    face_result = verify_face(driver_id, verbose=verbose)

    log_event(
        "FACE_PASS" if face_result["passed"] else "FACE_FAIL",
        driver_id,
        face_result["passed"],
        {"score": face_result.get("score"), "error": face_result.get("error")}
    )

    if face_result["passed"]:
        token = _create_and_register_token(driver_id, "face", verbose)
        return token

    # ── LOCKOUT ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'═'*55}")
        print(f"  ⛔ ALL AUTH LEVELS FAILED — LOCKOUT")
        print(f"  System locked. Attempting again will require restart.")
        print(f"  {'═'*55}\n")

    log_event("LOCKOUT", driver_id, False,
              {"reason": "All 3 auth levels failed"})

    return None


def _create_and_register_token(driver_id: str, method: str, verbose: bool) -> str:
    """Helper: create session token, register it, log it."""
    token = create_token(driver_id, method)
    register_token(token)
    log_event("SESSION_START", driver_id, True,
              {"auth_method": method, "duration": 900})

    if verbose:
        print(f"\n  {'═'*55}")
        print(f"  ✅ IDENTITY CONFIRMED via {method.upper()}")
        print(f"  Session active — 15 minutes")
        print(f"  {'═'*55}\n")

    return token


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK SESSION — Called by Layer 7 for every command
# ══════════════════════════════════════════════════════════════════════════════

def check_session(token: str) -> dict:
    """
    Check if session token is still valid.

    Returns:
    {
        "valid":      True/False,
        "driver_id":  "driver1",
        "expires_in": 420,
        "error":      None
    }

    Layer 7 calls this before every command to decide
    whether to re-authenticate or proceed.
    """
    return check_token(token)


# ══════════════════════════════════════════════════════════════════════════════
#  PAYMENT AUTHORIZATION
# ══════════════════════════════════════════════════════════════════════════════

def authorize_payment(
    driver_id: str,
    token:     str,
    amount:    float,
    merchant:  str,
    verbose:   bool = True
) -> dict:
    """
    Full payment authorization flow.

    Checks session → re-verify if expired → Voice OTP → confirm

    Returns:
    {
        "authorized": True/False,
        "reason":     "Payment authorized" or "OTP failed" etc.
        "driver_id":  "driver1"
    }
    """
    if verbose:
        print(f"\n  {'═'*55}")
        print("  💳 PAYMENT AUTHORIZATION")
        print(f"  Amount   : Rs {amount}")
        print(f"  Merchant : {merchant}")
        print(f"  {'═'*55}")

    # Check session
    session = check_session(token)

    if not session["valid"]:
        if verbose:
            print("\n  ⚠️  Session expired. Re-verifying at payment threshold (0.92)...")

        # Re-verify at stricter threshold
        result = verify_for_payment(driver_id, verbose=verbose)

        log_event(
            "VOICE_PASS" if result["passed"] else "VOICE_FAIL",
            driver_id, result["passed"],
            {"context": "payment_reauth", "score": result["score"]}
        )

        if not result["passed"]:
            # Try PIN for payment
            pin_result = prompt_and_verify_pin(driver_id, verbose=verbose)
            if not pin_result["passed"]:
                log_event("PAYMENT_DENY", driver_id, False,
                          {"reason": "Auth failed", "amount": amount})
                return {"authorized": False, "reason": "Identity not verified",
                        "driver_id": driver_id}

    # Voice OTP
    otp_result = run_voice_otp(driver_id, verbose=verbose)

    if not otp_result["passed"]:
        log_event("PAYMENT_DENY", driver_id, False,
                  {"reason": "OTP failed", "amount": amount, "merchant": merchant})
        return {"authorized": False, "reason": "OTP failed", "driver_id": driver_id}

    # Final confirmation
    if verbose:
        print(f"\n  Nova says: \"Confirm: {merchant}, Rs {amount}. Say YES to proceed.\"")
    confirm = input("  You: ").strip().lower()

    if confirm not in ["yes", "yeah", "yep", "confirm", "ok", "okay"]:
        log_event("PAYMENT_DENY", driver_id, False,
                  {"reason": "User cancelled", "amount": amount})
        return {"authorized": False, "reason": "Cancelled by user", "driver_id": driver_id}

    # Authorized
    log_event("PAYMENT_AUTH", driver_id, True,
              {"amount": amount, "merchant": merchant})

    if verbose:
        print(f"\n  ✅ PAYMENT AUTHORIZED")
        print(f"  Rs {amount} → {merchant}")
        print(f"  Transaction digitally signed and logged.\n")

    return {"authorized": True, "reason": "Payment authorized", "driver_id": driver_id}


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Full demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NOVA Layer 3 — Full Auth Flow")
    parser.add_argument("--driver",   type=str,  default="driver1")
    parser.add_argument("--payment",  action="store_true", help="Test payment flow")
    parser.add_argument("--show-log", action="store_true", help="Show audit log at end")
    args = parser.parse_args()

    # Run authentication
    token = authenticate(args.driver)

    if token is None:
        print("  System locked out. Exiting.")
        sys.exit(1)

    # If payment test requested
    if args.payment:
        result = authorize_payment(
            driver_id=args.driver,
            token=token,
            amount=180,
            merchant="Starbucks"
        )
        print(f"\n  Payment result: {result['reason']}")

    # Show audit log
    if args.show_log:
        print_log(args.driver)
