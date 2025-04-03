import os
import tempfile
import base64
import logging
from io import BytesIO

import speech_recognition as sr
from gtts import gTTS

logger = logging.getLogger(__name__)

def transcribe_audio(audio_file_path, recognizer=None, language='en-US'):
    """
    Transcribe audio file to text using the SpeechRecognition library.
    
    Parameters:
        audio_file_path (str): Path to the audio file.
        recognizer (sr.Recognizer, optional): A custom recognizer instance.
        language (str): Language code for transcription (default: 'en-US').
    
    Returns:
        str: Transcribed text.
    
    Raises:
        Exception: If transcription fails.
    """
    if recognizer is None:
        recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            logger.info("Performing speech recognition...")
            # Using Google's API for transcription; adjust as needed.
            text = recognizer.recognize_google(audio_data, language=language)
            logger.info(f"Transcription result: {text}")
            return text
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}", exc_info=True)
        raise e

def generate_speech_response(text, lang='en'):
    """
    Convert text into speech using gTTS and return the audio data as a base64 encoded string.
    
    Parameters:
        text (str): Text to convert into speech.
        lang (str): Language code for TTS (default: 'en').
    
    Returns:
        str: Base64 encoded audio string of the generated speech.
    
    Raises:
        Exception: If TTS conversion fails.
    """
    try:
        logger.info("Generating speech using gTTS...")
        tts = gTTS(text=text, lang=lang)
        tts_fp = BytesIO()
        tts.write_to_fp(tts_fp)
        tts_fp.seek(0)
        audio_base64 = base64.b64encode(tts_fp.read()).decode('utf-8')
        logger.info("Speech generation successful")
        return audio_base64
    except Exception as e:
        logger.error(f"Failed to generate speech response: {str(e)}", exc_info=True)
        raise e

def save_temp_audio_file(audio_file, suffix=".wav", temp_dir='temp'):
    """
    Save an uploaded audio file to a temporary location.
    
    Parameters:
        audio_file: A file-like object representing the uploaded audio.
        suffix (str): File suffix (default: '.wav').
        temp_dir (str): Directory to store temporary files.
    
    Returns:
        str: Path to the saved temporary file.
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=temp_dir)
    audio_file.save(temp_file.name)
    temp_file.close()
    logger.info(f"Saved temporary audio file: {temp_file.name}")
    return temp_file.name
