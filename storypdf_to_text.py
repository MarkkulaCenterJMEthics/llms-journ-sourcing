#!/usr/bin/env python3
"""
Process PDF files and save content in article extractor format.
Reads PDF files from command line arguments.
"""

import os
import re
import sys
import argparse
from datetime import datetime

# Try to import PDF libraries
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_LIBRARY = None

def check_dependencies():
    """Check if required PDF libraries are installed."""
    if PDF_LIBRARY is None:
        print("Error: No PDF processing library found.")
        print("Please install one of the following:")
        print("  pip install PyPDF2")
        print("  pip install pdfplumber")
        print("\nRecommended: pip install pdfplumber (better text extraction)")
        sys.exit(1)
    else:
        print(f"Using {PDF_LIBRARY} for PDF processing")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file using available library.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        if PDF_LIBRARY == "pdfplumber":
            return extract_with_pdfplumber(pdf_path)
        elif PDF_LIBRARY == "PyPDF2":
            return extract_with_pypdf2(pdf_path)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return None

def extract_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber (recommended)."""
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_with_pypdf2(pdf_path):
    """Extract text using PyPDF2."""
    import PyPDF2
    
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

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
    text = re.sub(r'[<>:"/\\|?*\n\r\t\'"`\u2019]', '_', text)
    
    # Replace multiple spaces/underscores with single underscore
    text = re.sub(r'[\s_]+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    # Limit length and ensure it's not empty
    text = text[:max_length] if text else "untitled"
    
    return text.lower()

def clean_pdf_text(text):
    """
    Clean up common PDF extraction artifacts.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\d{1,3}/\d{1,3}\n', '\n', text)  # Page numbers like "1/6"
    text = re.sub(r'\n\d{1,2}/\d{1,2}/\d{2,4}.*?\n', '\n', text)  # Date headers
    
    # Remove excessive spaces
    text = re.sub(r' {3,}', ' ', text)
    
    # Clean up line breaks around URLs
    text = re.sub(r'\nhttps?://[^\s]+\s*\n', '\n', text)
    
    return text.strip()

def extract_metadata_from_text(text, pdf_filename):
    """
    Extract title, author, and date from PDF text content.
    
    Args:
        text (str): PDF text content
        pdf_filename (str): Original PDF filename
        
    Returns:
        dict: Metadata dictionary
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    metadata = {
        'title': None,
        'author': None,
        'date': None,
        'url': None
    }
    
    # Extract title (usually one of the first substantial lines)
    for i, line in enumerate(lines[:10]):
        # Skip short lines, bylines, dates, and common headers
        if (len(line) > 15 and 
            not line.startswith(('By ', 'Listen to', 'Reporting from', 'April ', 'January ', 'February ', 'March ', 'May ', 'June ', 'July ', 'August ', 'September ', 'October ', 'November ', 'December ')) and
            not re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}', line) and
            not line.lower().startswith(('copyright', 'all rights', 'page \d+'))):
            metadata['title'] = line
            break
    
    # Extract author (look for "By Author Name" pattern)
    for line in lines:
        if line.startswith('By '):
            # Clean up author line
            author = line.replace('By ', '').strip()
            # Remove extra info after author name
            author = re.split(r'\n|Reporting from|has covered', author)[0].strip()
            metadata['author'] = author
            break
    
    # Extract date (look for date patterns)
    date_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            metadata['date'] = match.group()
            break
    
    # Extract URL (look for nytimes.com links)
    url_match = re.search(r'https://www\.nytimes\.com/[^\s]+', text)
    if url_match:
        metadata['url'] = url_match.group()
    
    # Fallback title from filename if not found
    if not metadata['title']:
        metadata['title'] = os.path.splitext(pdf_filename)[0].replace('_', ' ').title()
    
    return metadata

def save_article(title, content, author=None, date=None, source_url=None, source_file=None, output_folder="2025_extracted_articles"):
    """
    Save article content in the standard format.
    
    Args:
        title (str): Article title
        content (str): Article content
        author (str): Author name
        date (str): Publication date
        source_url (str): Original URL
        source_file (str): Source PDF filename
        output_folder (str): Output directory
    
    Returns:
        str: Path of saved file
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Generate filename
    clean_title = clean_filename(title)
    output_file = f"{clean_title}.txt"
    
    # Ensure unique filename
    counter = 1
    base_filename = clean_title
    while os.path.exists(os.path.join(output_folder, output_file)):
        output_file = f"{base_filename}_{counter}.txt"
        counter += 1
    
    # Write content to file
    file_path = os.path.join(output_folder, output_file)
    with open(file_path, 'w', encoding='utf-8') as file:
        # Write metadata header
        file.write("=" * 50 + "\n")
        file.write(f"Title: {title}\n")
        if author:
            file.write(f"Author: {author}\n")
        if date:
            file.write(f"Date: {date}\n")
        if source_url:
            file.write(f"URL: {source_url}\n")
        if source_file:
            file.write(f"Source PDF: {source_file}\n")
        file.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("=" * 50 + "\n\n")
        
        # Write main content
        file.write(content)
    
    return file_path

def process_pdf_file(pdf_path, output_folder="2025_extracted_articles"):
    """
    Process a single PDF file and save as article.
    
    Args:
        pdf_path (str): Path to PDF file
        output_folder (str): Output directory
        
    Returns:
        str: Path of saved file, or None if failed
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    print(f"Processing: {pdf_path}")
    
    # Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print(f"Failed to extract text from: {pdf_path}")
        return None
    
    # Clean up the text
    cleaned_text = clean_pdf_text(raw_text)
    
    # Extract metadata
    pdf_filename = os.path.basename(pdf_path)
    metadata = extract_metadata_from_text(cleaned_text, pdf_filename)
    
    print(f"  Title: {metadata['title']}")
    if metadata['author']:
        print(f"  Author: {metadata['author']}")
    if metadata['date']:
        print(f"  Date: {metadata['date']}")
    
    # Save article
    saved_path = save_article(
        title=metadata['title'],
        content=cleaned_text,
        author=metadata['author'],
        date=metadata['date'],
        source_url=metadata['url'],
        source_file=pdf_filename,
        output_folder=output_folder
    )
    
    return saved_path

def main():
    """Main function to handle command line arguments and process PDFs."""
    parser = argparse.ArgumentParser(
        description="Extract article content from PDF files and save with auto-generated filenames",
        epilog="Example: python pdf_processor.py article1.pdf article2.pdf"
    )
    
    parser.add_argument(
        "pdf_files",
        nargs="+",
        help="PDF files to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="2025_extracted_articles",
        help="Output folder for saved articles (default: 2025_extracted_articles)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install recommended PDF processing library (pdfplumber)"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing pdfplumber...")
        os.system("pip install pdfplumber")
        return
    
    # Check dependencies
    check_dependencies()
    
    # Process each PDF file
    successful = 0
    failed = 0
    
    for pdf_path in args.pdf_files:
        result = process_pdf_file(pdf_path, args.output)
        if result:
            print(f"✓ Saved: {result}")
            successful += 1
        else:
            print(f"❌ Failed: {pdf_path}")
            failed += 1
        print()  # Empty line between files
    
    # Summary
    print(f"🎉 Processing complete!")
    print(f"✓ Successful: {successful}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print(f"Files saved in: {args.output}/")

if __name__ == "__main__":
    main()