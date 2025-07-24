import os
import zipfile
import requests

# URL of the zip file
zip_url = 'https://github.com/samuel-oluwemimo/Datasets/raw/refs/heads/main/gopalkalpande_bbc_news_summary_raw-20250624T111156Z-1-001.zip'
local_zip_file = 'bbc_news_summary.zip'
extract_dir = 'bbc_news_summary'

# Download the zip file
response = requests.get(zip_url)
with open(local_zip_file, 'wb') as f:
    f.write(response.content)

# Create directory and unzip
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Downloaded and extracted successfully.")
