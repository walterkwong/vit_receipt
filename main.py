import torch
import pandas as pd
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
)
from PIL import Image

from image_process import ensure_rgb
from extraction import extract_total_amount
from extraction import (
    extract_receipt_date,
    categorize_goods,
    extract_payee,
)
import os
from pathlib import Path
import easygui

def prompt_folder():
    try:
        folder_selected = easygui.diropenbox("Select the folder with receipt images here:")
        return folder_selected
    except ImportError:
        folder_selected = input("Enter the path to the folder with receipt images: ").strip()
        return folder_selected

def main():
    folder_selected = prompt_folder()
    if not folder_selected or not Path(folder_selected).is_dir():
        print("Invalid or no folder selected.")
        return

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    files = [
        os.path.join(folder_selected, f)
        for f in os.listdir(folder_selected)
        if f.lower().endswith(image_extensions)
    ]

    if not files:
        print("No image files found in the selected folder.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/paligemma2-3b-pt-224"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config
    ).eval()

    processor = PaliGemmaProcessor.from_pretrained(model_id)

    results = {"Date": [], "Payee": [], "Total Amount": [], "Category": [], "File": []}

    for file_path in files:
        image = Image.open(file_path)
        image = ensure_rgb(image)

        date = extract_receipt_date(image, device, model, processor)
        payee = extract_payee(image, device, model, processor)
        total_amount = extract_total_amount(image, device, model, processor)
        category = categorize_goods(image, device, model, processor)

        results["Date"].append(date if date else "N/A")
        results["Payee"].append(payee if payee else "N/A")
        results["Total Amount"].append(total_amount if total_amount else 0.0)
        results["Category"].append(category if category else "Other")
        results["File"].append(os.path.basename(file_path))

    # Optionally, convert results to a DataFrame and save
    df = pd.DataFrame(results)
    output_path = os.path.join(folder_selected, "output.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
