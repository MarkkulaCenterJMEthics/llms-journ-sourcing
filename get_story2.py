#!/usr/bin/env python3
"""
Article extractor that fetches web content and saves it with an auto-generated filename
based on the page title.
"""

import trafilatura
import argparse
import os
import re
import sys

def clean_filename(text, max_length=50):
    """
    Clean text to make it suitable for use as a filename.
    
    Args:
        text (str): The text to clean
        max_length (int): Maximum length for the filename
    
    Returns:
        str: Cleaned filename-safe text
    """
    if not text:
        return "untitled"
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace problematic characters with underscores
    text = re.sub(r'[<>:"/\\|?*\n\r\t\'"`]', '_', text)

    
    # Replace multiple spaces/underscores with single underscore
    text = re.sub(r'[\s_]+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    # Limit length and ensure it's not empty
    text = text[:max_length] if text else "untitled"
    
    return text.lower()

def extract_article_with_links(url, output_folder="extracted_articles"):
    """
    Extract article content from URL and save with auto-generated filename.
    
    Args:
        url (str): URL to extract content from
        output_folder (str): Folder to save the extracted content
    
    Returns:
        str: Path of the saved file, or None if extraction failed
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    try:
        # Fetch content from the URL
        print(f"Fetching content from: {url}")
        
        # Try with custom headers to mimic a real browser
        config = trafilatura.settings.use_config()
        config.set("DEFAULT", "USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            print(f"Failed to fetch content from {url}. Check if the URL is valid and accessible.")
            return None
        
        # Extract content including links
        content = trafilatura.extract(downloaded, include_links=False)
        
        # Extract metadata (including title)
        metadata = trafilatura.extract_metadata(downloaded)
        
        if not content:
            print(f"Failed to extract readable content from {url}.")
            return None
        
        # Generate filename based on title
        if metadata and metadata.title:
            title = metadata.title
            print(f"Page title: {title}")
        else:
            # Fallback: try to extract title from content
            lines = content.split('\n')
            title = lines[0] if lines else "untitled"
            print(f"Using first line as title: {title}")
        
        # Clean title for filename
        clean_title = clean_filename(title)
        output_file = f"{clean_title}.txt"
        
        # Ensure unique filename if file already exists
        counter = 1
        base_filename = clean_title
        while os.path.exists(os.path.join(output_folder, output_file)):
            output_file = f"{base_filename}_{counter}.txt"
            counter += 1
        
        # Write content to file
        file_path = os.path.join(output_folder, output_file)
        with open(file_path, 'w', encoding='utf-8') as file:
            # Write metadata as header
            file.write("=" * 50 + "\n")
            if metadata:
                if metadata.title:
                    file.write(f"Title: {metadata.title}\n")
                if metadata.author:
                    file.write(f"Author: {metadata.author}\n")
                if metadata.date:
                    file.write(f"Date: {metadata.date}\n")
            file.write(f"URL: {url}\n")
            file.write("=" * 50 + "\n\n")
            
            # Write the main content
            file.write(content)
        
        print(f"✓ Article saved successfully: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def main():
    """Main function to handle command line arguments and execute extraction."""
    parser = argparse.ArgumentParser(
        description="Extract article content from a URL and save it with auto-generated filename",
        epilog="Example: python get_story2.py https://example.com/article"
    )
    
    parser.add_argument(
        "url", 
        help="URL of the article to extract"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="2025_extracted_articles",
        help="Output folder for saved articles (default: 2025_extracted_articles)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies (trafilatura)"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing trafilatura...")
        os.system("pip install trafilatura")
        return
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: Please provide a valid URL starting with http:// or https://")
        sys.exit(1)
    
    # Extract article
    result = extract_article_with_links(args.url, args.output)
    
    if result:
        print(f"\n🎉 Extraction completed successfully!")
        print(f"File saved at: {result}")
    else:
        print(f"\n❌ Extraction failed for: {args.url}")
        sys.exit(1)

if __name__ == "__main__":
    main()