# Claude Code — Project Context

## About the user

- Background: biology + Spanish, minimal CS experience.
- Needs step-by-step explanations, not just solutions.
- Prefers beginner-friendly advice and is learning Python alongside this
  project.

## About the project

A **Spanish ↔ English speech translator** meant as an **assistive tool**
for volunteer, non-certified interpreters at free clinics. Not a
replacement for a certified medical interpreter.

**Design principles, in order:**

1. **Safety** — the human interpreter must always be able to see the
   Spanish transcription and the English translation before anything is
   spoken. Never hide errors.
2. **Offline-first** — clinics have poor wifi, and real patient data
   must not go to third-party APIs. Prefer local models.
3. **Simplicity** — the user is learning. Prefer clear, readable code
   over clever code. Explain what each change does.
4. **Cultural relevance** — Spanish is not monolithic. Vocabulary and
   folk-illness concepts vary by country (Mexican, Caribbean, Central
   American, Andean, etc.). Future work will add a variant-aware
   terminology layer.

## Stack

- **faster-whisper** for ASR (runs locally, CPU or GPU)
- **NLLB-200 distilled 600M** from Hugging Face for MT (local)
- **pyttsx3** for offline TTS
- Python 3.10+, managed via a local `venv`

## Current state

Baseline pipeline works in `translator.py`:
- `--text` mode: translate a Spanish string to English
- `--file` mode: translate a Spanish .wav file
- interactive mode: bidirectional microphone loop

## How to help

When the user asks for help:

- **Diagnose before installing.** If a command fails, read the full
  error before reaching for `pip install` again.
- **Prefer `venv`** over global installs or conda unless the user has
  a reason to use something else.
- **On macOS Apple Silicon**, faster-whisper works on CPU out of the
  box; don't try to force CUDA.
- **On Linux**, remember pyttsx3 needs `espeak` and sounddevice needs
  `portaudio19-dev`.
- **Never edit `translator.py` silently.** Describe the change, then
  apply it, then explain what it does.
- **Before suggesting new dependencies**, check if the problem can be
  solved with what's already installed.

## Things to avoid

- Do not propose cloud-API solutions (OpenAI, Google, Azure) for the
  core translation path. The offline constraint is intentional.
- Do not remove the safety-framing comments at the top of
  `translator.py` or the README.
- Do not silently swap NLLB for a different translation model without
  discussing trade-offs first.

## Good next steps to suggest, when the baseline works

1. Logging of transcription + translation to a file.
2. Cultural concept flagger for folk-illness terms.
3. Medication / number preservation pass.
4. Confidence threshold that asks the speaker to repeat.
5. Voice-activity detection instead of fixed-duration recording.
6. Upgrade to Coqui TTS for more natural voice.
