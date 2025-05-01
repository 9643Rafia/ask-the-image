from gtts import gTTS
import tempfile

def generate_speech(answer):
    tts = gTTS(answer)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    return temp_path
