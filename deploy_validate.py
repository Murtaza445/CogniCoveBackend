"""Pre-flight validation for deployment.

Checks that model files are present, have realistic sizes,
and are NOT Git LFS pointer files.
Run this during CI/build or at startup.
"""

import os
import sys

# (path, minimum_size_mb)
REQUIRED_MODELS = [
    ("models/speech_emotion/model.safetensors", 100),
    ("models/facial_emotion/best_fer_model.pth", 1),
    (
        "Crisis Detection Model/electra_suicidal_text_detector/electra_suicidal_text_detector/model.safetensors",
        100,
    ),
    ("models/piper/en_US-lessac-medium.onnx", 10),
    ("models/vosk-model/am/final.mdl", 1),
]


def validate() -> bool:
    """Return True if all model files look valid."""
    errors = []
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for rel_path, min_size_mb in REQUIRED_MODELS:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            errors.append(f"MISSING: {rel_path}")
            continue

        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        if size_mb < min_size_mb:
            errors.append(f"TOO SMALL ({size_mb:.1f} MB < {min_size_mb} MB): {rel_path}")

        with open(full_path, "rb") as f:
            header = f.read(128)
            if b"version https://git-lfs.github.com/spec" in header:
                errors.append(f"GIT LFS POINTER (not real weights): {rel_path}")

    if errors:
        print("[FAIL] DEPLOYMENT VALIDATION FAILED")
        for msg in errors:
            print(f"   → {msg}")
        print()
        print("   Fix: run 'git lfs pull' or download the real model weights.")
        return False

    print("[PASS] Deployment validation passed — all model files look valid.")
    return True


if __name__ == "__main__":
    ok = validate()
    if not ok and "--fail-on-error" in sys.argv:
        sys.exit(1)
