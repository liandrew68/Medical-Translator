# Medical Spanish ↔ English Speech Translator (Starter)

A beginner-friendly, **assistive** speech translator for volunteer
interpreters at free clinics. Records from the microphone, transcribes
Spanish or English, translates, and speaks the result back.

> ⚠️ **Not a replacement for a certified medical interpreter.** Do not
> use with real patient data without addressing HIPAA, clinical-accuracy
> evaluation, and deployment controls. This repo is a learning starter.

## What's in this folder

| File               | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| `translator.py`    | The main script (mic loop + file mode + text mode)     |
| `app.py`           | Web interface (Streamlit app)                         |
| `requirements.txt` | Python dependencies                                    |
| `README.md`        | This file                                              |
| `CLAUDE.md`        | Context for Claude Code when you hand off the project  |
| `diagnose_audio.py`| Audio troubleshooting and testing tool                 |

## Deploy Online (for Non-Technical Users)

The easiest way for non-technical users to access this app is to deploy it online using **Streamlit Cloud**. This creates a web app that anyone can use with just a browser - no installation required!

### Quick Deploy to Streamlit Cloud

1. **Create a GitHub repository**:
   ```bash
   # Create a new repo on GitHub.com
   # Upload these files: app.py, translator.py, requirements.txt, packages.txt
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Share the URL**: Anyone can now access the translator at the generated URL!

### Alternative: One-Click Installer

For users who prefer a local app, you can create a simple installer script:

```bash
# Create install.sh
echo '#!/bin/bash
echo "Installing Medical Translator..."
python3 -m venv translator_env
source translator_env/bin/activate
pip install -r requirements.txt
echo "Installation complete! Run with: source translator_env/bin/activate && streamlit run app.py"
' > install.sh
chmod +x install.sh
```

Users just run `./install.sh` and then `streamlit run app.py`.

---

## How the pipeline works

```
┌──────────┐   ┌──────────────────┐   ┌────────────┐   ┌──────┐
│ Mic in   │ → │ faster-whisper   │ → │  NLLB-200  │ → │ TTS  │ → Speakers
│ (audio)  │   │ (speech → text)  │   │ (translate)│   │      │
└──────────┘   └──────────────────┘   └────────────┘   └──────┘
```

All three stages run **locally on your laptop**. No cloud, no API keys.
First run downloads ~3 GB of model weights; after that, it's offline.

---

## Quick start with Claude Code (recommended)

Since you're new to CS, the easiest path is to let Claude Code handle
the environment setup and error debugging for you.

### 1. Install Claude Code

Follow the instructions at https://docs.claude.com/en/docs/claude-code

### 2. Open this folder in Claude Code

```
cd path/to/translator-project
claude
```

### 3. Paste this first message to Claude Code

> I'm a beginner at coding (background: biology + Spanish). I want to
> run this Spanish-English medical speech translator. Please:
>
> 1. Check that Python 3.10+ is installed (install if needed, using the
>    approach that's safest for my system).
> 2. Create a virtual environment in this folder called `venv` and
>    activate it.
> 3. Install everything in `requirements.txt`.
> 4. On Linux, also make sure `espeak` is installed (pyttsx3 needs it).
> 5. Run a quick self-test: `python translator.py --text "Me duele la
>    cabeza"` and confirm you see an English translation.
> 6. If that works, help me test the microphone mode by running
>    `python translator.py` and walking me through one Spanish-to-English
>    turn.
>
> Explain each step briefly before running it, and if anything fails,
> diagnose the error and try the next reasonable fix. Don't skip
> explanations — I want to learn.

Claude Code will walk you through the rest.

---

## Manual setup (if you prefer)

### Prerequisites

- **Python 3.10 or newer** (check with `python3 --version`)
- **~5 GB free disk space** for model downloads
- **A working microphone** (for interactive mode)
- **RAM**: 8 GB minimum, 16 GB recommended
- **OS-specific**:
  - **macOS**: nothing extra needed
  - **Windows**: nothing extra needed
  - **Linux**: `sudo apt install espeak portaudio19-dev` (for TTS + mic)

### Install

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows

# Install dependencies (this is the slow step)
pip install -r requirements.txt
```

### Run

```bash
# Easiest test first — translate a Spanish string, no mic/audio involved
python translator.py --text "Me duele la cabeza desde hace tres días"

# Translate a pre-recorded Spanish audio file
python translator.py --file my_recording.wav

# Full interactive mode with microphone
python translator.py

# Web interface (recommended for easy use)
streamlit run app.py
# Then open http://localhost:8501 in your browser
```

The web app provides a user-friendly interface with four translation modes:
- Spanish Audio → English Speech
- Spanish Text → English Speech  
- English Audio → Spanish Speech
- English Text → Spanish Speech

---

## Troubleshooting

| Problem                                   | Likely fix                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named ...`| You forgot to activate the virtual environment. Run `source venv/bin/activate`. |
| First run hangs for minutes               | Normal — it's downloading ~3 GB of models. Give it 5–15 min on first run.   |
| No sound from TTS on Linux                | `sudo apt install espeak`                                                   |
| `PortAudioError` / mic not found          | Linux: `sudo apt install portaudio19-dev`, then `pip install --force-reinstall sounddevice` |
| Out of memory                             | Change `WHISPER_MODEL_SIZE = "small"` to `"base"` or `"tiny"` in `translator.py` |
| Translation is slow (>10 s per sentence)  | Expected on CPU with `medium` or larger. Use `small` for responsive demos.  |
| Caribbean Spanish mis-transcribed         | Known Whisper weakness — try `medium` or `large-v3` if your machine can handle it |

---

## What to try once it works

Short list, roughly in order:

1. **Log everything.** Add a transcription + translation log to a file
   so you can review what the system heard vs. said.
2. **Cultural concept flagging.** Add a dictionary of folk-illness and
   region-specific terms (`empacho`, `susto`, `nervios`, `monga`,
   `pasmo`, `bicho` varying by country, etc.). Before translating,
   check the Spanish transcription for these and surface a note to the
   interpreter.
3. **Number / medication safety.** Pre-process numbers and drug names
   so they pass through the translator unchanged; post-verify they
   appear intact in the output.
4. **Confidence threshold.** faster-whisper exposes per-segment
   probability. When low, prompt the user to repeat rather than
   proceeding silently.
5. **Better TTS voice.** Swap pyttsx3 for Coqui TTS (local, nicer
   voice) or edge-tts (requires internet, very natural).
6. **Voice-activity detection.** Replace fixed-duration recording
   with start/stop on speech using `webrtcvad` or `silero-vad`.

---

## Safety reminder

This system will mistranslate. The whole design assumption is that a
human interpreter watches the transcription + translation text on
screen, catches errors, and corrects them before they reach the patient
or clinician. Do not remove that human from the loop.
