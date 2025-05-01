
import gradio as gr
from asr import transcribe_audio
from qa import generate_answer
from tts import generate_speech
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def process(image, audio):
    try:
        # Handle image format and convert
        image = image.convert("RGB")
    except Exception as e:
        return f"Uploaded image is invalid. Please try again.", None

    # Transcribe audio
    try:
        question = transcribe_audio(audio, device)
    except Exception as e:
        return f"Error transcribing audio: {e}", None

    # Generate image caption and answer
    try:
        image_caption, answer = generate_answer(image, question, device)
    except Exception as e:
        return f"Error answering question: {e}", None

    # Text-to-speech
    try:
        audio_path = generate_speech(answer)
    except Exception as e:
        return f"Error generating TTS: {e}", None

    return f"Image Caption: {image_caption}\nQ: {question}\nA: {answer}", audio_path

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Ask the Image (Speech + Vision + Text)")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        audio_input = gr.Audio(type="filepath", label="Ask a Question (mic or upload)")

    output_text = gr.Textbox(label="Transcribed Q, Image Caption, and Answer")
    output_audio = gr.Audio(label="Audio Answer")

    submit_btn = gr.Button("Process")
    submit_btn.click(fn=process, inputs=[image_input, audio_input], outputs=[output_text, output_audio])

demo.launch(debug=True)
