#!/usr/bin/env python3
"""
Streamlit web interface for the Medical Spanish-English Speech Translator.

Usage:
    streamlit run app.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import re
from scipy.io import wavfile

# Import translator components
from translator import (
    load_all_models,
    transcribe,
    translate,
    speak,
    TRANSCRIPTION_CORRECTIONS,
    TRANSLATION_RULES,
    POST_TRANSLATION_RULES,
)


def apply_translation_rules(text, rules=None):
    """Apply custom word/phrase substitutions to translated text."""
    if rules is None:
        rules = TRANSLATION_RULES
    result = text
    for spanish_word, english_word in rules.items():
        pattern = re.compile(r'\b' + re.escape(spanish_word) + r'\b', re.IGNORECASE)
        result = pattern.sub(english_word, result)
    
    for original_phrase, replacement_phrase in POST_TRANSLATION_RULES.items():
        pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
        result = pattern.sub(replacement_phrase, result)
    
    return result


def transcribe_with_corrections(whisper, audio_or_path, language):
    """Transcribe and apply corrections."""
    segments, _info = whisper.transcribe(
        audio_or_path,
        language=language,
        beam_size=5,
        vad_filter=False,
        no_speech_threshold=0.2,
    )
    segments = list(segments)

    text_parts = []
    for seg in segments:
        corrected_text = seg.text
        for misheard, correct in TRANSCRIPTION_CORRECTIONS.items():
            pattern = re.compile(r'\b' + re.escape(misheard) + r'\b', re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
        text_parts.append(corrected_text)

    return " ".join(text_parts).strip()


# Page config
st.set_page_config(
    page_title="Medical Spanish-English Translator",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 Medical Spanish-English Speech Translator")
st.markdown(
    "**⚠️ ASSISTIVE TOOL ONLY** — Not a replacement for certified medical interpreters. "
    "Use for learning and support only."
)

# Load models (cached)
@st.cache_resource
def load_models():
    return load_all_models()

with st.spinner("Loading AI models (first run takes ~2 min)..."):
    whisper, tokenizer, nllb, tts = load_models()

# Sidebar: Translation rules editor
with st.sidebar:
    st.header("⚙️ Custom Translation Rules")
    st.subheader("Spanish → English word mappings")
    
    # Display current rules
    st.write("**Current Rules:**")
    for spanish, english in TRANSLATION_RULES.items():
        st.write(f"  • `{spanish}` → `{english}`")
    
    st.info(
        "Edit `TRANSLATION_RULES` in `translator.py` to add more custom mappings. "
        "Then restart this app."
    )

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["📁 Spanish Audio → English", "📁 English Audio → Spanish", "📝 Spanish to English", "📝 English to Spanish"])

with tab1:
    st.subheader("Upload Spanish Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3", "m4a"],
        help="Upload a Spanish audio file to translate to English.",
        key="spanish_audio",
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("🔄 Translate Audio", key="translate_spanish_audio"):
            with st.spinner("Processing..."):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    # Transcribe
                    st.info("Transcribing audio...")
                    transcription = transcribe_with_corrections(whisper, tmp_path, "es")
                    st.success(f"**Heard:** {transcription}")
                    
                    # Translate
                    st.info("Translating...")
                    translation = translate(tokenizer, nllb, transcription, "spa_Latn", "eng_Latn")
                    translation = apply_translation_rules(translation)
                    st.success(f"**Translation:** {translation}")
                    
                    # Speak
                    st.info("Generating speech...")
                    speak(tts, translation)
                    st.success("✅ Done! Audio played.")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Spanish (Original):**")
                        st.text(transcription)
                    with col2:
                        st.write("**English (Translation):**")
                        st.text(translation)
                
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

with tab2:
    st.subheader("Upload English Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3", "m4a"],
        help="Upload an English audio file to translate to Spanish.",
        key="english_audio",
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("🔄 Translate Audio", key="translate_english_audio"):
            with st.spinner("Processing..."):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    # Transcribe
                    st.info("Transcribing audio...")
                    transcription = transcribe_with_corrections(whisper, tmp_path, "en")
                    st.success(f"**Heard:** {transcription}")
                    
                    # Translate
                    st.info("Translating...")
                    translation = translate(tokenizer, nllb, transcription, "eng_Latn", "spa_Latn")
                    st.success(f"**Translation:** {translation}")
                    
                    # Speak
                    st.info("Generating speech...")
                    speak(tts, translation)
                    st.success("✅ Done! Audio played.")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**English (Original):**")
                        st.text(transcription)
                    with col2:
                        st.write("**Spanish (Translation):**")
                        st.text(translation)
                
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

with tab3:
    st.subheader("Translate Spanish Text to English")
    
    spanish_text = st.text_area(
        "Enter Spanish text to translate",
        placeholder="e.g., 'Me duele la cabeza mucho. Estoy chambeando.'",
        height=100,
        key="spanish_input",
    )
    
    if st.button("🔄 Translate to English", key="translate_spanish"):
        if spanish_text.strip():
            with st.spinner("Translating..."):
                # Translate
                translation = translate(
                    tokenizer, nllb, spanish_text, "spa_Latn", "eng_Latn"
                )
                translation = apply_translation_rules(translation)
                
                st.success(f"**Translation:** {translation}")
                
                # Speak
                if st.button("🔊 Speak English", key="speak_english"):
                    with st.spinner("Speaking..."):
                        speak(tts, translation)
                        st.success("✅ Audio played.")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Spanish:**")
                    st.text(spanish_text)
                with col2:
                    st.write("**English:**")
                    st.text(translation)
        else:
            st.warning("Please enter Spanish text to translate.")

with tab4:
    st.subheader("Translate English Text to Spanish")
    
    english_text = st.text_area(
        "Enter English text to translate",
        placeholder="e.g., 'I have a severe headache that started three days ago.'",
        height=100,
        key="english_input",
    )
    
    if st.button("🔄 Translate to Spanish", key="translate_english"):
        if english_text.strip():
            with st.spinner("Translating..."):
                # Translate
                translation = translate(
                    tokenizer, nllb, english_text, "eng_Latn", "spa_Latn"
                )
                
                st.success(f"**Translation:** {translation}")
                
                # Speak
                if st.button("🔊 Speak Spanish", key="speak_spanish"):
                    with st.spinner("Speaking..."):
                        speak(tts, translation)
                        st.success("✅ Audio played.")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**English:**")
                    st.text(english_text)
                with col2:
                    st.write("**Spanish:**")
                    st.text(translation)
        else:
            st.warning("Please enter English text to translate.")

# Footer
st.markdown("---")
st.markdown(
    "Built with [Streamlit](https://streamlit.io), "
    "[Whisper](https://openai.com/research/whisper), "
    "[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M), "
    "and [pyttsx3](https://pyttsx3.readthedocs.io)"
)
