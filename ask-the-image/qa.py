import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def generate_answer(image, question, device="cpu"):
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    image_inputs = blip_processor(image, return_tensors="pt").to(device)
    image_output = blip_model.generate(**image_inputs)
    image_caption = blip_processor.tokenizer.decode(image_output[0], skip_special_tokens=True)
    
    full_prompt = f"Caption: {image_caption}. Now answer the following question: {question}"
    inputs = blip_processor(image, text=full_prompt, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    answer = blip_processor.tokenizer.decode(output[0], skip_special_tokens=True)
    
    return image_caption, answer
