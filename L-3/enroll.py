"""
enroll.py
---------
Registers a new driver into the NOVA system.

What it does:
  1. Records 5 voice samples from the driver
  2. Extracts voice fingerprint using ECAPA-TDNN (SpeechBrain)
  3. Averages 5 fingerprints into 1 master fingerprint
  4. Encrypts and saves the fingerprint (AES-256)
  5. Captures a face photo from webcam
  6. Extracts face embedding and saves encrypted
  7. Takes a PIN and saves it as SHA-256 hash

Run this ONCE per driver:
  python enroll.py --driver driver1
"""

import os
import sys
import time
import hashlib
import argparse
import numpy as np
from pathlib import Path

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from crypto_utils import save_array, encrypt_and_save

# ── Data folders ───────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
VOICEPRINT_DIR = DATA_DIR / "voiceprints"
FACE_DIR       = DATA_DIR / "faces"
PIN_DIR        = DATA_DIR / "pins"

for d in [VOICEPRINT_DIR, FACE_DIR, PIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — RECORD VOICE SAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from microphone.
    Returns numpy array of audio samples.
    duration: how many seconds to record
    """
    import pyaudio

    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    total_chunks = int(sample_rate / CHUNK * duration)

    for i in range(total_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.float32))

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)


def preprocess_audio(pcm_f32: np.ndarray) -> np.ndarray:
    """
    Match the preprocessing applied by stt_moonshine_worker before ECAPA-TDNN embedding.
    Without this, enrollment embeddings are built from raw mic audio while verification
    embeddings come from pre-emphasized + RMS-normalized audio, causing ~0.04 cosine
    similarity drop that pushes scores below the 0.6 threshold.
    """
    _preemph_coeff = 0.97
    _rms_target    = 0.08
    _rms_floor     = 1e-6

    # Pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
    out = np.empty_like(pcm_f32)
    out[0] = pcm_f32[0]
    out[1:] = pcm_f32[1:] - _preemph_coeff * pcm_f32[:-1]

    # RMS normalization
    rms = np.sqrt(np.mean(out ** 2))
    if rms > _rms_floor:
        out = out * (_rms_target / rms)

    return out


def save_wav(audio: np.ndarray, filepath: Path, sample_rate: int = 16000) -> None:
    """Save numpy audio array as a WAV file."""
    import soundfile as sf
    sf.write(str(filepath), audio, sample_rate)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — EXTRACT VOICE FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_fingerprint(wav_path: Path) -> np.ndarray:
    """
    Extract a 192-dimensional voice fingerprint from a WAV file.
    Uses ECAPA-TDNN model from SpeechBrain.
    First call downloads ~80MB model. Subsequent calls are instant.

    Returns: numpy array of shape (192,)
    """
    from speechbrain.inference.speaker import EncoderClassifier

    print("    [fingerprint] Loading ECAPA-TDNN model...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    import torch
    import torchaudio

    # Load audio
    waveform, sr = torchaudio.load(str(wav_path))

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Extract embedding
    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    # Returns shape (1, 1, 192) — flatten to (192,)
    fingerprint = embedding.squeeze().cpu().numpy()
    return fingerprint


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — ENROLL VOICE
# ══════════════════════════════════════════════════════════════════════════════

def enroll_voice(driver_id: str, num_samples: int = 5) -> bool:
    """
    Records num_samples voice samples, extracts fingerprints,
    averages them into one master fingerprint, encrypts and saves.
    """
    print(f"\n{'='*55}")
    print(f"  VOICE ENROLLMENT — {driver_id}")
    print(f"{'='*55}")
    print(f"  We will record {num_samples} voice samples.")
    print(f"  Speak naturally for 3 seconds each time.")
    print(f"  Tip: vary slightly — normal, morning voice, relaxed")
    print(f"{'='*55}\n")

    # Temp folder for raw WAV files
    temp_dir = DATA_DIR / "temp_enrollment"
    temp_dir.mkdir(exist_ok=True)

    fingerprints = []

    for i in range(1, num_samples + 1):
        input(f"  Sample {i}/{num_samples} — Press ENTER then speak for 3 seconds...")
        print(f"  🎙️  Recording... speak now")

        audio = record_audio(duration=3.0)
        audio = preprocess_audio(audio)
        wav_path = temp_dir / f"{driver_id}_sample_{i}.wav"
        save_wav(audio, wav_path)

        print(f"  ✅  Recorded. Extracting fingerprint...")
        try:
            fp = extract_fingerprint(wav_path)
            fingerprints.append(fp)
            print(f"  📊  Fingerprint extracted (shape: {fp.shape})")
        except Exception as e:
            print(f"  ❌  Error extracting fingerprint: {e}")
            return False

    # Average all fingerprints into one master
    master_fingerprint = np.mean(fingerprints, axis=0)
    print(f"\n  🔄  Averaging {num_samples} fingerprints into master...")

    # Normalize (important for cosine similarity later)
    master_fingerprint = master_fingerprint / np.linalg.norm(master_fingerprint)

    # Encrypt and save
    save_path = VOICEPRINT_DIR / f"{driver_id}.enc"
    save_array(master_fingerprint, save_path)

    print(f"  🔐  Encrypted voiceprint saved → {save_path}")
    print(f"  ✅  Voice enrollment complete!\n")

    # Cleanup temp files
    for f in temp_dir.glob(f"{driver_id}_*.wav"):
        f.unlink()

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ENROLL FACE
# ══════════════════════════════════════════════════════════════════════════════

def enroll_face(driver_id: str, interactive: bool = True) -> bool:
    """
    Captures a face photo from webcam, extracts face embedding,
    encrypts and saves it.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force DeepFace/TensorFlow to use CPU
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    import cv2

    print(f"\n{'='*55}")
    print(f"  FACE ENROLLMENT — {driver_id}")
    print(f"{'='*55}")
    print(f"  We will capture your face using the webcam.")
    print(f"{'='*55}\n")

    if interactive:
        input("  Press ENTER to activate webcam. Look directly at camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ❌  Webcam not found. Skipping face enrollment.")
        print("      Face ID fallback will not be available.")
        return False

    print("  📷  Webcam active. Capturing in 3 seconds...")
    time.sleep(3)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  ❌  Failed to capture frame.")
        return False

    # Save temp photo
    temp_path = DATA_DIR / f"temp_{driver_id}_face.jpg"
    cv2.imwrite(str(temp_path), frame)
    print(f"  ✅  Photo captured.")

    # Extract face embedding using DeepFace
    try:
        from deepface import DeepFace
        import json

        print("  🔄  Extracting face embedding...")
        result = DeepFace.represent(
            img_path=str(temp_path),
            model_name="Facenet",
            enforce_detection=True
        )
        embedding = np.array(result[0]["embedding"], dtype=np.float32)
        print(f"  📊  Face embedding extracted (shape: {embedding.shape})")

        # Encrypt and save
        save_path = FACE_DIR / f"{driver_id}.enc"
        save_array(embedding, save_path)
        print(f"  🔐  Encrypted face embedding saved → {save_path}")
        print(f"  ✅  Face enrollment complete!\n")

    except Exception as e:
        print(f"  ❌  Face extraction failed: {e}")
        print(f"      Make sure your face is clearly visible.")
        temp_path.unlink(missing_ok=True)
        return False

    # Cleanup temp photo
    temp_path.unlink(missing_ok=True)
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — ENROLL PIN
# ══════════════════════════════════════════════════════════════════════════════

def enroll_pin(driver_id: str, pin: str = None) -> bool:
    """
    Takes a PIN from the driver (typed for enrollment, spoken later),
    SHA-256 hashes it, and saves the hash.
    The actual PIN is NEVER stored — only the hash.
    """
    print(f"\n{'='*55}")
    print(f"  PIN ENROLLMENT — {driver_id}")
    print(f"{'='*55}")
    print(f"  Set a 4-digit PIN as fallback if voice fails.")
    print(f"  The PIN is NEVER stored — only a hash of it.")
    print(f"{'='*55}\n")

    import getpass

    if pin is None:
        while True:
            pin = getpass.getpass("  Enter 4-digit PIN (hidden): ")
            if not pin.isdigit() or len(pin) != 4:
                print("  ❌  PIN must be exactly 4 digits. Try again.")
                continue

            confirm = getpass.getpass("  Confirm PIN: ")
            if pin != confirm:
                print("  ❌  PINs don't match. Try again.")
                continue

            break

    # Hash with SHA-256
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()

    # Save hash
    save_path = PIN_DIR / f"{driver_id}.txt"
    with open(save_path, "w") as f:
        f.write(pin_hash)

    print(f"\n  🔐  PIN hash saved → {save_path}")
    print(f"  ✅  PIN enrollment complete!\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENROLLMENT FLOW
# ══════════════════════════════════════════════════════════════════════════════

def enroll_driver(driver_id: str, skip_face: bool = False) -> None:
    """
    Full enrollment flow for a new driver.
    Runs voice → face → PIN enrollment in sequence.
    """
    print(f"\n{'#'*55}")
    print(f"#  NOVA — DRIVER ENROLLMENT")
    print(f"#  Driver ID: {driver_id}")
    print(f"{'#'*55}")

    # Check if already enrolled
    voice_path = VOICEPRINT_DIR / f"{driver_id}.enc"
    if voice_path.exists():
        overwrite = input(f"\n  ⚠️  Driver '{driver_id}' already enrolled. Re-enroll? (y/n): ")
        if overwrite.lower() != 'y':
            print("  Enrollment cancelled.")
            return

    results = {}

    # 1. Voice enrollment
    results["voice"] = enroll_voice(driver_id)

    # 2. Face enrollment
    if not skip_face:
        results["face"] = enroll_face(driver_id)
    else:
        print("\n  [Face enrollment skipped]")
        results["face"] = False

    # 3. PIN enrollment
    results["pin"] = enroll_pin(driver_id)

    # Summary
    print(f"\n{'='*55}")
    print(f"  ENROLLMENT SUMMARY — {driver_id}")
    print(f"{'='*55}")
    print(f"  Voice fingerprint : {'✅ Done' if results['voice'] else '❌ Failed'}")
    print(f"  Face ID           : {'✅ Done' if results['face'] else '⚠️  Skipped/Failed'}")
    print(f"  PIN               : {'✅ Done' if results['pin'] else '❌ Failed'}")
    print(f"{'='*55}")

    if results["voice"] and results["pin"]:
        print(f"\n  🎉  Driver '{driver_id}' enrolled successfully!")
        print(f"  You can now run verify.py to test identity verification.\n")
    else:
        print(f"\n  ⚠️  Enrollment incomplete. Voice and PIN are required.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOVA Driver Enrollment")
    parser.add_argument("--driver", type=str, default="driver1",
                        help="Driver ID (default: driver1)")
    parser.add_argument("--skip-face", action="store_true",
                        help="Skip face enrollment (use if no webcam)")
    args = parser.parse_args()

    enroll_driver(args.driver, skip_face=args.skip_face)
