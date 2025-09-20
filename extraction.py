from PIL import Image
import re
from inference import ask_model
from datetime import datetime
from typing import Optional
    
def extract_receipt_date(image: Image.Image, model, tokenizer, processor) -> Optional[datetime.date]:
    '''
    Extracts the date from a receipt image using the provided model, tokenizer, and processor.
    Input: PIL Image, model, tokenizer, processor
    Output: datetime.date object or None if not found
    '''
    
    question = ("what is the date of the receipt? please only answer in YYYY-MM-DD format.")
    answer = ask_model(image, question, model, tokenizer, processor)
    print(f"Date from model: {answer}")
    match = re.search(r"\d{4}-\d{2}-\d{2}", answer)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None

def extract_payee(image: Image.Image, model, tokenizer, processor) -> str:
    '''
    Extracts the payee from a receipt image using the provided model, tokenizer, and processor.
    Input: PIL Image, model, tokenizer, processor
    Output: Payee as a string
    '''

    question = "who is this payment going to? the answer should be a company name only. Do not answer anything else other than the name."
    answer = ask_model(image, question, model, tokenizer, processor)
    print(f"Payee from model: {answer}")
    answer = answer.split("\n")[-1].strip().capitalize()
    return answer


def extract_total_cost(image: Image.Image, model, tokenizer, processor) -> Optional[float]:
    '''
    Extracts the total cost from a receipt image using the provided model, tokenizer, and processor.
    Input: PIL Image, model, tokenizer, processor
    Output: Total amount as a float or None if not found
    '''

    question = "what is the total cost in the receipt? answer in numbers only."
    answer = ask_model(image, question, model, tokenizer, processor)
    print(f"Answer from model: {answer}")
    try:
        amounts = re.findall(
            r"\d+\.\d+|\d+", answer
        )  # Capture both integer and decimal values
        if amounts:
            return float(amounts[-1])  # Get the last valid amount as the total
    except ValueError:
        pass
    return None


def categorize_goods(image: Image.Image, model, tokenizer, processor) -> str:
    '''
    Categorizes the goods/services in a receipt image using the provided model, tokenizer, and processor.
    Input: PIL Image, model, tokenizer, processor
    Output: Category as a string
    '''

    question = "what is the categorise the goods/services in the receipt? answer 1 for Entertainment/Food/luxurious Goods, 2 for Office Utility/Equipments/Tools, 3 for Insurance, 4 for Lab, 5 for Mandatory Provident Fund (MPF), 6 for Charity, 7 for Rent. Do not answer anything else."
    answer = ask_model(image, question, model, tokenizer, processor)
    print(f"Category from model: {answer}")
    answer = answer.split("\n")[-1].strip().capitalize()
    valid_categories = {
        "1": "Entertainment",
        "2": "Office Utility",
        "3": "Insurance",
        "4": "Lab",
        "5": "Mandatory Provident Fund (MPF)",
        "6": "Charity",
        "7": "Rent",
    }
    if answer in valid_categories:
        return valid_categories[answer]
    else:
        return "Other"
