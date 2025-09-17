from PIL import Image
import re
from inference import ask_model
from datetime import datetime


from typing import Optional

def extract_receipt_date(
    image: Image.Image, device, model, processor
) -> Optional[datetime.date]:
    question = "What is the date of the receipt? Answer in YYYY-MM-DD format."
    answer = ask_model(image, question, device, model, processor)
    print(f"Date from model: {answer}")
    match = re.search(r"\d{4}-\d{2}-\d{2}", answer)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def extract_payee(image: Image.Image, device, model, processor) -> str:
    question = (
        "Who is this payment going to? Answer is a company name, in English or Chinese."
    )
    answer = ask_model(image, question, device, model, processor)
    print(f"Payee from model: {answer}")
    answer = answer.split("\n")[-1].strip().capitalize()
    return answer


def extract_total_amount(image: Image.Image, device, model, processor) -> Optional[float]:
    question = "ocr\n what is the total amount of money used? in numbers only"
    answer = ask_model(image, question, device, model, processor)
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


def categorize_goods(image: Image.Image) -> str:
    question = "What is the category of goods or services in the image? Answer with the best option as listed below: Entertainment/Office Utility/Insurance/Lab/Mandatory Provident Fund (MPF)/Charity/Rent. Do not answer anything else."
    answer = ask_model(image, question)
    print(f"Category from model: {answer}")
    answer = answer.split("\n")[-1].strip().capitalize()
    valid_categories = [
        "Entertainment",
        "Office Utility",
        "Insurance",
        "Lab",
        "Mandatory Provident Fund (MPF)",
        "Charity",
        "Rent",
    ]
    if answer in valid_categories:
        return answer
    else:
        return "Other"
