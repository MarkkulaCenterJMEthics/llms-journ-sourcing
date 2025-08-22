!pip install trafilatura
import trafilatura
import requests
from bs4 import BeautifulSoup
import trafilatura
import os
import trafilatura

def extract_article_with_links(url, output_folder, output_file):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch and extract content from the URL, including hyperlinks
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded, include_links=True)

    # Write content to the output file within the folder
    if content:
        file_path = os.path.join(output_folder, output_file)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"File saved successfully in {file_path}")
    else:
        print(f"Failed to extract content from {url}. Check if the URL is valid and accessible.")

# Usage
url = 'https://calmatters.org/california-divide/2024/01/border-patrol-california/'
output_folder = 'Extracted_Articles_links'  # Folder name you want to create
output_file = 'calmatters_borderpatrol_links.txt'
extract_article_with_links(url, output_folder, output_file)