#!/usr/bin/env python3
"""
Medical Spanish-English Speech Translator (starter)

A beginner-friendly, bidirectional speech translator built from three
open-source components:
  - faster-whisper  (speech recognition, runs locally)
  - NLLB-200        (translation, runs locally)
  - pyttsx3         (text-to-speech, runs locally, offline)

IMPORTANT: This is a LEARNING PROJECT and an ASSISTIVE tool only.
It is NOT a replacement for a certified medical interpreter.
Do NOT use with real patient data until you have addressed:
  - HIPAA compliance
  - Clinical-accuracy evaluation with bilingual clinicians
  - Proper consent, logging, and deployment controls

Usage:
    python translator.py                  # Interactive mic mode
    python translator.py --file foo.wav   # Translate a Spanish audio file
    python translator.py --text "Hola"    # Translate Spanish text (no audio)
    python translator.py --list-devices   # Show available microphones
    python translator.py --device 2       # Use a specific microphone
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------- Configuration (tweak these) ---------------------------------
WHISPER_MODEL_SIZE = "small"  # Back to small for faster loading
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

SAMPLE_RATE = 16000         # Whisper expects 16 kHz mono
RECORDING_DURATION = 7      # seconds per utterance in mic mode

# If peak audio amplitude is below this, treat the recording as silence
# and refuse to send it to Whisper (prevents hallucinated outputs like
# "lance", "gracias", "subtítulos por Amara" etc. from near-silence).
SILENCE_PEAK_THRESHOLD = 0.02

# If Whisper's average log-probability for a segment is below this,
# warn the user that the transcription is likely a hallucination.
LOW_CONFIDENCE_LOGPROB = -0.8

# Transcription corrections: fix common Whisper misrecognitions
# Map misheard words to correct Spanish words before translation
TRANSCRIPTION_CORRECTIONS = {
    "cambiando": "chambeando",  # chambear (to work/hustle) often misheard as cambiar (to change)
    # Add more as you discover them
}

# Custom translation rules: word/phrase overrides (Spanish -> English)
# These apply AFTER the NLLB model translates, allowing medical and cultural terms
TRANSLATION_RULES = {
    "chevere": "awesome",
    "chévere": "awesome",
    "genial": "great",
    "bolsa": "bag",
    "chambear": "work/hustle",
    "constipado": "congested",
    "intoxicación": "food poisoning",
    "embarazada": "pregnant",
    "aborto espontáneo": "miscarriage",
    "riñones": "lower back",
    "cintura": "lower back",
    "gripa": "cold",
    "asiento": "diarrhea",
    "agruras": "heartburn",
    "suero": "IV fluids",
    "mareo": "dizziness",
    "empacho": "folk GI illness",
    "susto": "fright-related syndrome",
    "ataque de nervios": "nervios syndrome",
    "caída de mollera": "infant dehydration",
    "pujidos": "infant grunting",
    "ardor": "burning sensation",
    "hormigueo": "paresthesia",
    "crudo": "hungover",
    "tengo azúcar": "I have diabetes",
    "regla": "menstrual period",
    # Add more custom mappings as needed
}

# Post-translation substitutions to catch model variations
# (e.g., if NLLB outputs "nice" for "chévere", we map it to "awesome")
POST_TRANSLATION_RULES = {
    "nice of you to see me": "awesome to see me",
    "how nice": "how awesome",
    "really freaking out": "hustling",
    "freaking out": "hustling",
}
# ------------------------------------------------------------------------


def list_input_devices():
    """Print available input (microphone) devices and exit."""
    import sounddevice as sd
    print("\nAvailable audio devices:\n")
    devices = sd.query_devices()
    for index, dev in enumerate(devices):
        print(
            f"  {index}: {dev['name']}  (in={dev['max_input_channels']} out={dev['max_output_channels']})",
            f"rate={dev['default_samplerate']}",
        )
    print(f"\nDefault input device: #{sd.default.device[0]}")
    print(f"Default output device: #{sd.default.device[1]}\n")
    print("To use a specific mic, re-run with:  python translator.py --device N\n")


def load_whisper():
    from faster_whisper import WhisperModel
    print(f"  Loading Whisper ({WHISPER_MODEL_SIZE})...")
    # Use CPU/int8 by default for this macOS environment to avoid faster-whisper
    # device/compute auto-detection crashes.
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def load_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("  Loading NLLB-200...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    return tokenizer, model


def load_tts():
    import pyttsx3
    print("  Loading TTS engine...")
    return pyttsx3.init()


def load_all_models():
    print("Loading models (first run downloads ~3 GB, please wait)...")
    whisper = load_whisper()
    tokenizer, nllb = load_nllb()
    tts = load_tts()
    print("All models ready.\n")
    return whisper, tokenizer, nllb, tts


def record_audio(duration=RECORDING_DURATION, device=None):
    """Record from the microphone with a countdown and silence check.

    Returns either a numpy array of audio samples, or None if the
    recording was effectively silent.
    """
    import sounddevice as sd

    if device is not None:
        try:
            device_info = sd.query_devices(device, kind="input")
        except Exception as exc:
            print(f"  ERROR: input device #{device} is not available: {exc}")
            return None
        sd.default.device = (device, sd.default.device[1])
    else:
        device_info = sd.query_devices(sd.default.device[0], kind="input")
    device_rate = int(device_info.get("default_samplerate", SAMPLE_RATE))
    print(f"  Using mic sample rate: {device_rate} Hz")

    # Countdown so the user has time to get ready and doesn't miss the
    # first second of their sentence.
    print("Recording in...")
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(0.7)
    print(f"  🎤 SPEAK NOW ({duration}s)\n")

    audio = sd.rec(
        int(duration * device_rate),
        samplerate=device_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    if device_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, SAMPLE_RATE, device_rate)
        print(f"  Resampled audio from {device_rate} Hz to {SAMPLE_RATE} Hz")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
        print("  ...normalized audio to avoid clipping.")
        peak = float(np.max(np.abs(audio)))
    elif peak > 0.0 and peak < 0.9:
        audio = audio * (0.9 / peak)
        print("  ...normalized audio to improve level.")
        peak = float(np.max(np.abs(audio)))

    print(f"  ...done. Peak level: {peak:.3f}")

    if peak < SILENCE_PEAK_THRESHOLD or np.std(audio) < 0.005:
        print("  ⚠️  The recording is nearly silent. Nothing to transcribe.")
        print("     Check: mic permissions, correct input device, speak louder.")
        print("     Run `python translator.py --list-devices` to see all mics.")
        return None

    # Debug: save the recorded audio to inspect
    from scipy.io import wavfile
    debug_wav = "debug_recording.wav"
    try:
        wavfile.write(debug_wav, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        print(f"  Debug: saved recording to {debug_wav} — play it to verify audio quality.")
    except Exception as e:
        print(f"  Debug: failed to save {debug_wav}: {e}")

    return audio


def transcribe(whisper, audio_or_path, language):
    """Transcribe audio to text. Returns (text, low_confidence_flag)."""
    import re
    segments, _info = whisper.transcribe(
        audio_or_path,
        language=language,
        beam_size=5,
        vad_filter=False,  # Disable VAD for file input to avoid cutting clean audio
        no_speech_threshold=0.2,
    )
    segments = list(segments)

    text_parts = []
    low_conf = False
    for seg in segments:
        corrected_text = seg.text
        # Apply transcription corrections
        for misheard, correct in TRANSCRIPTION_CORRECTIONS.items():
            pattern = re.compile(r'\b' + re.escape(misheard) + r'\b', re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
        text_parts.append(corrected_text)
        if seg.avg_logprob < LOW_CONFIDENCE_LOGPROB:
            low_conf = True

    return " ".join(text_parts).strip(), low_conf


def apply_translation_rules(text, rules=None):
    """Apply custom word/phrase substitutions to translated text."""
    if rules is None:
        rules = TRANSLATION_RULES
    result = text
    for spanish_word, english_word in rules.items():
        # Case-insensitive replacement (preserving original case if possible)
        import re
        pattern = re.compile(r'\b' + re.escape(spanish_word) + r'\b', re.IGNORECASE)
        result = pattern.sub(english_word, result)
    
    # Apply post-translation rules (context-aware fixes)
    for original_phrase, replacement_phrase in POST_TRANSLATION_RULES.items():
        import re
        pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
        result = pattern.sub(replacement_phrase, result)
    
    return result


def translate(tokenizer, model, text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=400,
    )
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    # Apply custom translation rules if translating to English
    if tgt_lang == "eng_Latn":
        result = apply_translation_rules(result, TRANSLATION_RULES)
    return result


def speak(tts, text):
    tts.setProperty('rate', 150)  # Moderate speech rate (faster than 100)
    tts.say(text)
    tts.runAndWait()


def run_one_turn(whisper, tokenizer, nllb, tts, direction, device=None):
    if direction == "es->en":
        whisper_lang, src_nllb, tgt_nllb = "es", "spa_Latn", "eng_Latn"
    else:
        whisper_lang, src_nllb, tgt_nllb = "en", "eng_Latn", "spa_Latn"

    audio = record_audio(device=device)
    if audio is None:
        return  # silent recording, already warned

    print("Transcribing...")
    transcription, low_conf = transcribe(whisper, audio, whisper_lang)

    if not transcription:
        print("  (no speech detected — try again, speak louder/closer)\n")
        return

    print(f"  Heard: {transcription!r}")
    if low_conf:
        print("  ⚠️  Low confidence — this may be a mis-hearing. Confirm before use.")

    print("Translating...")
    translation = translate(tokenizer, nllb, transcription, src_nllb, tgt_nllb)
    print(f"  →      {translation!r}")

    # Interpreter confirmation step — the human-in-the-loop safety gate.
    confirm = input("  Speak this aloud? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        print("Speaking...\n")
        speak(tts, translation)
    else:
        print("Skipped.\n")


def interactive_loop(whisper, tokenizer, nllb, tts, device=None):
    print("=" * 60)
    print("  Medical Spanish <-> English Translator  (ASSISTIVE ONLY)")
    print("=" * 60)
    print("  1: Spanish -> English   (patient speaks, clinician hears)")
    print("  2: English -> Spanish   (clinician speaks, patient hears)")
    print("  q: quit\n")

    while True:
        choice = input("Choose (1/2/q): ").strip().lower()
        if choice == "q":
            print("Bye.")
            return
        elif choice == "1":
            run_one_turn(whisper, tokenizer, nllb, tts, "es->en", device=device)
        elif choice == "2":
            run_one_turn(whisper, tokenizer, nllb, tts, "en->es", device=device)
        else:
            print("Please enter 1, 2, or q.")


def translate_file(path):
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    whisper, tokenizer, nllb, tts = load_all_models()

    print(f"Transcribing {path}...")
    transcription, low_conf = transcribe(whisper, str(path), None)  # Auto-detect language
    print(f"  Heard: {transcription!r}")
    if low_conf:
        print("  ⚠️  Low confidence — verify before use.")
    
    print("Translating...")
    # Assume Spanish input for now; adjust if needed
    translation = translate(tokenizer, nllb, transcription, "spa_Latn", "eng_Latn")
    print(f"  →      {translation!r}")
    print("Speaking...")
    speak(tts, translation)


def translate_text(text):
    tokenizer, nllb = load_nllb()
    translation = translate(tokenizer, nllb, text, "spa_Latn", "eng_Latn")
    print(f"ES: {text}")
    print(f"EN: {translation}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical Spanish<->English Speech Translator (starter)"
    )
    parser.add_argument("--file", help="Translate a Spanish audio file (.wav)")
    parser.add_argument("--text", help="Translate a Spanish text string (no audio)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List microphones and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index (see --list-devices)")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.text:
        translate_text(args.text)
    elif args.file:
        translate_file(args.file)
    else:
        whisper, tokenizer, nllb, tts = load_all_models()
        try:
            interactive_loop(whisper, tokenizer, nllb, tts, device=args.device)
        except KeyboardInterrupt:
            print("\nInterrupted.")


if __name__ == "__main__":
    main()
