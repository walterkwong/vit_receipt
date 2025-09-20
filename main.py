import torch
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from image_process import ensure_rgb, image_enhance
from extraction import extract_total_cost
from extraction import (
    extract_receipt_date,
    categorize_goods,
    extract_payee,
)
from transformers import AutoModel, AutoTokenizer, AutoProcessor

import easygui
import time

def prompt_folder():
    '''
    Prompt the user to select a folder using a GUI dialog. If easygui is not available, fall back to console input.
    '''

    try:
        folder_selected = easygui.diropenbox("Select the folder with receipt images here:")
        return folder_selected
    except ImportError:
        folder_selected = input("Enter the path to the folder with receipt images: ").strip()
        return folder_selected

def main():
    # Prompt user to select folder
    folder_selected = prompt_folder()
    if not folder_selected or not Path(folder_selected).is_dir():
        print("Invalid or no folder selected.")
        return

    # Gather image files from the selected folder
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    files = [
        os.path.join(folder_selected, f)
        for f in os.listdir(folder_selected)
        if f.lower().endswith(image_extensions)
    ]
    if not files:
        print("No image files found in the selected folder.")
        return
    files.sort() 

    # Load model, tokenizer, and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_id = "openbmb/MiniCPM-V-4_5-int4"

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, attn_implementation='sdpa', dtype=torch.bfloat16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    results = {"Date": [], "Payee": [], "Total Amount": [], "Category": [], "File": []}

    # Process each image file
    for file_path in files:
        print(f"\n\nProcessing file: {file_path}")
        start_time = time.time()

        image = Image.open(file_path)
        image = ensure_rgb(image)
        image = image_enhance(image)
        image = image.resize((640, 480), Image.LANCZOS)

        with torch.no_grad():
            date = extract_receipt_date(image, model, tokenizer, processor)
            payee = extract_payee(image, model, tokenizer, processor)
            total_amount = extract_total_cost(image, model, tokenizer, processor)
            category = categorize_goods(image, model, tokenizer, processor)

        results["Date"].append(date if date else "N/A")
        results["Payee"].append(payee if payee else "N/A")
        results["Total Amount"].append(total_amount if total_amount else 0.0)
        results["Category"].append(category if category else "Other")
        results["File"].append(os.path.basename(file_path))

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for {os.path.basename(file_path)}: {elapsed:.2f} seconds")

        # Clear cache after each iteration
        torch.cuda.empty_cache()
        del image

    # Save results to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(folder_selected, "output.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()