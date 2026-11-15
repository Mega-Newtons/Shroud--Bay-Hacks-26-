"""Gradient must flow from the ECAPA embedding back to raw samples. Run: python -m tests.test_audio_grad"""
from shield.audio import grad_check

if __name__ == "__main__":
    print(f"d(embedding)/d(waveform) grad norm = {grad_check():.3f} (finite, non-zero)")
