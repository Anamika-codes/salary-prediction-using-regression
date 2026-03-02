import os
import zipfile
import subprocess

# Dataset directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading dataset from Kaggle...")

# Kaggle dataset (Data Science Salaries)
subprocess.run([
    "kaggle",
    "datasets",
    "download",
    "-d",
    "arnabchaki/data-science-salaries-2023"
])

print("Extracting dataset...")

# Extract zip file
with zipfile.ZipFile("data-science-salaries-2023.zip", "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

# Remove zip after extraction
os.remove("data-science-salaries-2023.zip")

print("✅ Dataset downloaded and extracted inside /data folder")
