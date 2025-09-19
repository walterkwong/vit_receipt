# ViT Receipt Recorder

A Vision Transformer specialised in reading and recording receipts using MiniCPM-V-4_5-int4. Automatically processes all images in a folder, analyzes them, and generates a CSV file for easy auditing tasks.

GPU Highly Recommended.

## Installation

```bash
git clone https://github.com/walterkwong/vit_receipt
cd vit_receipt
pip install .
```

## Model Access
openbmb/MiniCPM-V-4_5-int4
1. Create an access token on Hugging Face.
2. Authenticate with your token:

    ```bash
    hf auth login
    ```

**Note:**  
The first run will download the model (~6.54 GB). Ensure you have sufficient disk space and a stable internet connection. Also, this is a very immature project solely for personal use, but do let me know if you want me to work on this project. 

Model used: openbmb/MiniCPM-V-4_5-int4