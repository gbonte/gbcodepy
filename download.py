from pathlib import Path
import requests
import zipfile

# Replace with your Dropbox direct-download link (dl=1)
URL = "https://www.dropbox.com/scl/fi/xxxxxxxx/data.zip?rlkey=yyyyyyyy&dl=1"

DATA_DIR = Path("Data")
ZIP_FILE = DATA_DIR / "data.zip"

# Create Data directory
DATA_DIR.mkdir(exist_ok=True)

# Download ZIP if needed
if not ZIP_FILE.exists():
    print("Downloading dataset...")

    with requests.get(URL, stream=True) as r:
        r.raise_for_status()

        with open(ZIP_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Download complete.")
else:
    print("ZIP file already exists.")

# Extract ZIP
print("Extracting files...")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print(f"Files extracted to: {DATA_DIR.resolve()}")