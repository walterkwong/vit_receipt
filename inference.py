from PIL import Image
    
def ask_model(image: Image.Image, question: str, model, tokenizer, processor, enable_thinking:bool = False, stream:bool = False) -> str:
    '''
    Asks a question to the model with the provided image and returns the model's answer.
    Input: PIL Image, question string, model, tokenizer, processor
    Output: Model's answer as a string
    '''

    prompt = f"{question}\n"
    msgs = [{'role': 'user', 'content': [image, prompt]}]
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=enable_thinking,
        stream=stream
    )
    return answer