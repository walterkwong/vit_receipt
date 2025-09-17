from PIL import Image
import torch


def ask_model(image: Image.Image, question: str, device, model, processor) -> str:
    prompt = f"<image> answer en {question}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return result[0].strip()
