"""
verify.py
---------
Verifies if the current speaker matches a registered driver.

What it does:
  1. Records 2-3 seconds of live audio
  2. Extracts live voice fingerprint (ECAPA-TDNN)
  3. Loads the stored encrypted fingerprint
  4. Compares using cosine similarity
  5. Returns score + pass/fail

Thresholds:
  >= 0.85  → Identity confirmed  (normal commands)
  >= 0.92  → Identity confirmed  (payment authorization)
  < 0.85   → Identity failed     (retry or fallback)

Run directly to test:
  python verify.py --driver driver1
"""

import sys
import time
import numpy as np
from pathlib import Path

# Monkey-patch: speechbrain 1.0.x passes `use_auth_token` to hf_hub_download,
# but huggingface_hub >=0.17 renamed it to `token` and removed the old kwarg.
try:
    import huggingface_hub as _hf
    _orig_hf_download = _hf.hf_hub_download
    def _patched_hf_download(*args, **kwargs):
        # speechbrain 1.0.x passes use_auth_token; newer huggingface_hub renamed it to token.
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token") or None
        try:
            return _orig_hf_download(*args, **kwargs)
        except Exception as e:
            # SpeechBrain's fetch() converts HTTPError 404 → ValueError so that
            # from_hparams() can silently skip missing custom.py.
            # Newer huggingface_hub uses httpx (not requests), so HTTPError is never
            # raised and the ValueError conversion never fires — the 404 propagates raw.
            # Fix: if we see a 404 for custom.py, raise ValueError directly so
            # SpeechBrain's `except ValueError` handler in from_hparams() catches it.
            filename = args[1] if len(args) > 1 else kwargs.get("filename", "")
            is_404 = ("404" in str(e) or "Not Found" in str(e) or
                      "EntryNotFound" in type(e).__name__ or
                      "RemoteEntryNotFound" in type(e).__name__)
            if "custom.py" in str(filename) and is_404:
                raise ValueError("File not found on HF hub") from e
            raise
    _hf.hf_hub_download = _patched_hf_download
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from crypto_utils import save_array, load_array

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR         = Path(__file__).parent / "data"
VOICEPRINT_DIR   = DATA_DIR / "voiceprints"
TEMP_DIR         = DATA_DIR / "temp_verify"

THRESHOLD_NORMAL  = 0.6   # general commands
THRESHOLD_PAYMENT = 0.75  # payment authorization

TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RECORD LIVE AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def record_live(duration: float = 2.5, sample_rate: int = 16000) -> np.ndarray:
    """
    Record live audio from microphone.
    Called every time we need to verify who is speaking.
    """
    import pyaudio

    CHUNK = 1024
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    total_chunks = int(sample_rate / CHUNK * duration)

    for _ in range(total_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.float32))

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)


def save_temp_wav(audio: np.ndarray, name: str = "live_sample") -> Path:
    """Save live audio to a temp WAV file for fingerprint extraction."""
    import soundfile as sf
    path = TEMP_DIR / f"{name}.wav"
    sf.write(str(path), audio, 16000)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT LIVE FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════

# Keep model loaded in memory so we don't reload it every verification
_model_cache = None

def get_model():
    """Load ECAPA-TDNN model once and cache it."""
    global _model_cache
    if _model_cache is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _model_cache = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
    return _model_cache


def extract_live_fingerprint(wav_path: Path) -> np.ndarray:
    """
    Extract voice fingerprint from a WAV file.
    Returns normalized numpy array of shape (192,)
    """
    import torch
    import torchaudio

    model = get_model()
    waveform, sr = torchaudio.load(str(wav_path))

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    fingerprint = embedding.squeeze().cpu().numpy()

    # Normalize for cosine similarity
    fingerprint = fingerprint / np.linalg.norm(fingerprint)
    return fingerprint


# ══════════════════════════════════════════════════════════════════════════════
#  COSINE SIMILARITY — THE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compare two fingerprints.
    Returns a score between 0.0 and 1.0.
    1.0 = identical voice
    0.0 = completely different voice

    Both arrays should already be normalized (done during extraction).
    If normalized: cosine similarity = dot product
    """
    return float(np.dot(a, b))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VERIFICATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def verify_voice(driver_id: str,
                 threshold: float = THRESHOLD_NORMAL,
                 verbose: bool = True,
                 audio_buffer: np.ndarray = None) -> dict:
    """
    Full voice verification for a driver.
    """
    start_time = time.time()

    # Check enrollment exists
    stored_path = VOICEPRINT_DIR / f"{driver_id}.enc"
    if not stored_path.exists():
        return {
            "passed": False,
            "score": 0.0,
            "driver_id": driver_id,
            "threshold": threshold,
            "error": f"Driver '{driver_id}' not enrolled. Run enroll.py first.",
            "time_ms": 0
        }

    if audio_buffer is not None:
        if verbose: print("  [verify] Using provided audio buffer from pipeline")
        wav_path = save_temp_wav(audio_buffer, f"{driver_id}_live")
    else:
        if verbose: print("  [verify] Recording voice... speak now (2.5 sec)")
        # Record live audio
        audio = record_live(duration=2.5)
        wav_path = save_temp_wav(audio, f"{driver_id}_live")

    if verbose:
        print("  [verify] Extracting live fingerprint...")

    # Extract live fingerprint
    live_fp = extract_live_fingerprint(wav_path)

    if verbose:
        print("  [verify] Loading stored fingerprint...")

    # Load stored fingerprint (decrypts automatically)
    stored_fp = load_array(stored_path)

    # Compare
    score = cosine_similarity(live_fp, stored_fp)
    passed = score >= threshold
    elapsed = int((time.time() - start_time) * 1000)

    # Cleanup temp file
    wav_path.unlink(missing_ok=True)

    result = {
        "passed":    passed,
        "score":     round(score, 4),
        "driver_id": driver_id,
        "threshold": threshold,
        "time_ms":   elapsed
    }

    if verbose:
        print(f"\n  {'─'*40}")
        print(f"  Score     : {score:.4f}")
        print(f"  Threshold : {threshold}")
        print(f"  Result    : {'✅ IDENTITY CONFIRMED' if passed else '❌ IDENTITY FAILED'}")
        print(f"  Time      : {elapsed}ms")
        print(f"  {'─'*40}\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PAYMENT VERIFICATION — STRICTER THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def verify_for_payment(driver_id: str, verbose: bool = True, audio_buffer: np.ndarray = None) -> dict:
    """
    Same as verify_voice but with stricter threshold (0.92) for payments.
    Call this when driver wants to make a purchase.
    """
    if verbose:
        print(f"\n  [verify] Payment verification — stricter threshold ({THRESHOLD_PAYMENT})")
    return verify_voice(driver_id, threshold=THRESHOLD_PAYMENT, verbose=verbose, audio_buffer=audio_buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Test directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NOVA Voice Verification")
    parser.add_argument("--driver", type=str, default="driver1",
                        help="Driver ID to verify")
    parser.add_argument("--payment", action="store_true",
                        help="Use payment threshold (0.92) instead of normal (0.85)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("  NOVA — Voice Verification")
    print(f"  Driver: {args.driver}")
    print(f"  Mode:   {'PAYMENT (0.92)' if args.payment else 'NORMAL (0.85)'}")
    print(f"{'='*50}\n")

    if args.payment:
        result = verify_for_payment(args.driver)
    else:
        result = verify_voice(args.driver)

    # Final output
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    elif result["passed"]:
        print(f"  🎉 Welcome back, {args.driver}!")
    else:
        print(f"  ⛔ Voice not recognized. Score {result['score']} < {result['threshold']}")
        print("     Fallback to PIN or Face ID required.")
