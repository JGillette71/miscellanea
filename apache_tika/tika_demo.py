
# System: Ubuntu 22.04.3 LTS
# Java Version: openjdk version "11.0.23" 2024-04-16
# Tika Version: 2.9.2
# Tesseract Version: tesseract 4.1.1
# See requirements.txt
# --------------------------------

# Ensure system packages are up to date
# sudo apt-get update

# Install Java (required for running Tika Server)
# sudo apt-get install default-jdk

# Install Tesseract OCR (optional, required for OCR functionality)
# sudo apt-get install tesseract-ocr-eng

# Download Apache Tika Server JAR
# wget https://downloads.apache.org/tika/2.9.2/tika-server-standard-2.9.2.jar -O tika-server-standard-2.9.2.jar

# Run Apache Tika Server with 2GB of memory on port 9999
# java -Xmx2g -jar tika-server-standard-2.9.2.jar -p 9999

# Verify Tika Server is running in a new terminal
# curl http://localhost:9999/version

import os
import requests
import json
from bs4 import BeautifulSoup

def get_file_paths(directory):
    """
    Returns a list of file paths for the specified directory.
    param: directory path where files stored
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def parse_file_with_tika(file_path):
    """
    Parses a file using Tika server and returns the parsed HTML/XHTML content.
    """
    print(f"Running parser for {file_path}...")
    url = 'http://localhost:9999/tika'
    headers = {"X-Tika-OCRLanguage": "eng"}
    with open(file_path, 'rb') as f:
        response = requests.put(url, headers=headers, data=f)
    response.raise_for_status()
    return response.text

def extract_required_fields(parsed_html):
    """
    Extracts the required fields from the parsed HTML content.
    """
    soup = BeautifulSoup(parsed_html, 'html.parser')
    metadata = {meta.get('name'): meta.get('content') for meta in soup.find_all('meta')}
    extracted_data = {
        "CreateDate": metadata.get("xmp:CreateDate"),
        "NPages": metadata.get("xmpTPg:NPages"),
        "ParsedBy": metadata.get("X-TIKA:Parsed-By"),
        "Content": soup.get_text()
    }
    return extracted_data

def save_to_json(all_extracted_data, output_file):
    """
    Saves all extracted data to a single JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, ensure_ascii=False, indent=4)

def main():
    data_dir = "./data"
    output_file = "./data/parsed_data.json"

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    file_paths = get_file_paths(data_dir)
    all_extracted_data = {}

    for file_path in file_paths:
        parsed_html = parse_file_with_tika(file_path)
        extracted_data = extract_required_fields(parsed_html)
        file_name = os.path.basename(file_path)
        all_extracted_data[file_name] = extracted_data

    save_to_json(all_extracted_data, output_file)
    print(f"Saved all parsed data to {output_file}")

if __name__ == "__main__":
    main()