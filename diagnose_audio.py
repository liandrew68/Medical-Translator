#!/usr/bin/env python3
"""
Audio diagnostic script.

Run this to figure out WHY the translator is only hearing silence/garbage.
It tests each stage independently:
  1. Lists all microphones your system sees.
  2. Records 5 seconds and reports how loud it was.
  3. Saves the recording to test_recording.wav so you can play it back.
  4. Runs Whisper on that file so you can see what it actually heard.

Usage:
    python diagnose_audio.py              # use default mic
    python diagnose_audio.py --device 2   # use microphone #2 from the list
    python diagnose_audio.py --duration 10  # record for 10 seconds instead of 5
"""

import argparse
import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000
OUTPUT_WAV = "test_recording.wav"


def list_devices():
    """Print all audio devices the system sees."""
    print("\n=== Audio devices on this system ===")
    devices = sd.query_devices()
    for index, dev in enumerate(devices):
        print(
            f"  {index}: {dev['name']}  (in {dev['max_input_channels']}, out {dev['max_output_channels']})",
            f"rate={dev['default_samplerate']}",
        )
    print("\nDefault input device:", sd.default.device[0])
    print("Default output device:", sd.default.device[1])
    print()


def record_and_analyze(duration, device=None):
    """Record audio and report loudness stats."""
    if device is not None:
        sd.default.device = (device, sd.default.device[1])
        print(f"Using input device #{device}: {sd.query_devices(device)['name']}")
    else:
        default_in = sd.default.device[0]
        print(f"Using default input device #{default_in}: "
              f"{sd.query_devices(default_in)['name']}")

    print(f"\n*** Recording for {duration} seconds. ***")
    print("*** SPEAK NOW, loudly and clearly, in Spanish. ***")
    print("*** Try: 'Me duele mucho la cabeza desde hace tres días.' ***\n")

    current_input = device if device is not None else sd.default.device[0]
    device_info = sd.query_devices(current_input)
    sample_rate = int(device_info.get("default_samplerate", SAMPLE_RATE))
    if device_info.get("max_input_channels", 0) < 1:
        raise ValueError(
            f"Selected device #{current_input} ('{device_info['name']}') does not support audio input."
        )

    print(f"  Recording at device sample rate: {sample_rate} Hz")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    # Loudness analysis
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    print("=== Recording results ===")
    print(f"  Samples captured:  {len(audio)}")
    print(f"  Peak amplitude:    {peak:.4f}  (0.0 = silence, 1.0 = max)")
    print(f"  RMS (avg loudness):{rms:.4f}")

    # Interpret
    if peak < 0.01:
        print("\n  ❌ DEAD SILENCE — the mic is not capturing anything.")
        print("     Check: mic permissions, correct input device, mic not muted.")
    elif peak < 0.05:
        print("\n  ⚠️  VERY QUIET — mic works but is picking up almost nothing.")
        print("     Fixes: speak louder, move closer, raise input volume in OS settings.")
    elif peak < 0.2:
        print("\n  ⚠️  QUIET — Whisper may still struggle.")
        print("     Try speaking louder or raising mic gain.")
    else:
        print("\n  ✅ Good levels. The mic is clearly capturing your voice.")

    # Save to WAV so user can play it back and verify
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(OUTPUT_WAV, sample_rate, audio_int16)
    print(f"\n  Saved to: {OUTPUT_WAV}")
    print("  👉 Play this file. If YOU can't hear your voice clearly, Whisper can't either.\n")

    return audio


def test_whisper_on_recording():
    """Run Whisper on the saved recording and print what it heard."""
    print("=== Running Whisper on the recording ===")
    from faster_whisper import WhisperModel

    print("Loading Whisper (small) on CPU...")
    # Use CPU/int8 for stability on macOS, where auto device selection can crash.
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Transcribing...")
    segments, info = model.transcribe(OUTPUT_WAV, language="es", beam_size=5)
    segments = list(segments)

    print(f"\n  Detected language probability: {info.language_probability:.2f}")
    print(f"  Duration:                      {info.duration:.2f}s")

    if not segments:
        print("\n  ❌ Whisper heard NOTHING meaningful.")
        print("     This matches what you saw ('lance' is Whisper's hallucination on silence).")
        return

    print("\n  Segments heard:")
    for seg in segments:
        # avg_logprob closer to 0 = more confident, more negative = less confident
        confidence_note = ""
        if seg.avg_logprob < -1.0:
            confidence_note = "  ⚠️ (very low confidence — likely hallucination)"
        elif seg.avg_logprob < -0.5:
            confidence_note = "  ⚠️ (low confidence)"
        print(f"    [{seg.start:.1f}s - {seg.end:.1f}s] "
              f"logprob={seg.avg_logprob:.2f}{confidence_note}")
        print(f"       → {seg.text!r}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose audio capture problems")
    parser.add_argument("--device", type=int, help="Input device index to use")
    parser.add_argument("--duration", type=int, default=5, help="Recording length in seconds")
    parser.add_argument("--list-only", action="store_true", help="Just list devices and exit")
    args = parser.parse_args()

    list_devices()

    if args.list_only:
        return

    input("Press ENTER when ready to record, then speak immediately...")
    record_and_analyze(args.duration, args.device)
    test_whisper_on_recording()


if __name__ == "__main__":
    main()
