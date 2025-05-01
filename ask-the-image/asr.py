import whisper

def transcribe_audio(audio_path, device="cpu"):
    whisper_model = whisper.load_model("small", device=device)
    result = whisper_model.transcribe(audio_path)
    return result["text"]
