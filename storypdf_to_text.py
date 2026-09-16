#!/usr/bin/env python3
"""
Convert a story PDF into extracted_articles_boilerplate/'s canonical
{story_number}-{slug}.txt format.

Unlike auto-detecting title/author/date from the PDF's own text (unreliable
-- easily fooled by browser-print headers, site nav chrome, and other
extraction noise), this script takes the story number and title as
explicit arguments, since by the time a PDF reaches this script we already
know both from the GT Expansion List. Author/date/url are optional.

Also strips two extraction artifacts common in PDFs made via a browser's
"print to PDF" function: a repeating "M/D/YY, H:MM AM/PM <title> | <site>"
header line at the top of every page, and a trailing "<url> <page>/<total>"
footer line at the bottom of every page. Anything beyond that (site nav
menus, "recommended articles" sections, comment threads, newsletter
signup forms, etc.) is site-specific and still needs a manual check --
this script does not attempt to strip it.
"""

import argparse
import os
import re
import sys

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber is required.\n  pip install pdfplumber", file=sys.stderr)
    sys.exit(1)

DATETIME_HEADER = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)\b')
URL_FOOTER = re.compile(r'^https?://\S+\s+\d+/\d+$')

DEFAULT_OUTPUT_FOLDER = "extracted_articles_boilerplate"


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF, page by page, stripping the repeating
    browser-print header/footer lines described above. Returns the
    cleaned full-text body (pages joined by a blank line), or None if no
    text could be extracted at all.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = [(p.extract_text() or "").strip() for p in pdf.pages]

    if not any(pages):
        return None

    cleaned_pages = []
    for page_text in pages:
        if not page_text:
            continue
        lines = page_text.split('\n')
        if lines and DATETIME_HEADER.match(lines[0]):
            lines = lines[1:]
        if lines and URL_FOOTER.match(lines[-1].strip()):
            lines = lines[:-1]
        cleaned = '\n'.join(lines).strip()
        if cleaned:
            cleaned_pages.append(cleaned)

    return '\n\n'.join(cleaned_pages) if cleaned_pages else None


def clean_slug(text, max_length=50):
    """Turn a title into the lowercase, underscore-separated slug used in
    extracted_articles_boilerplate/ filenames."""
    if not text:
        return "untitled"
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[<>:"/\\|?*\n\r\t\'’]', '_', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = text.strip('_')
    return text[:max_length].lower() if text else "untitled"


def save_article(story_number, title, body_text, author=None, date=None,
                  url=None, source_pdf=None, output_folder=DEFAULT_OUTPUT_FOLDER):
    """Write the article in the canonical Title/Author/Date/URL header
    format, at {output_folder}/{story_number}-{slug}.txt. Refuses to
    overwrite an existing file for that story number unprompted."""
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(
            f"Output folder '{output_folder}' does not exist -- refusing to "
            f"create it silently. Pass -o to point at the right corpus folder."
        )

    slug = clean_slug(title)
    filename = f"{story_number}-{slug}.txt"
    out_path = os.path.join(output_folder, filename)

    existing = [f for f in os.listdir(output_folder) if f.startswith(f"{story_number}-")]
    if existing and existing != [filename]:
        raise FileExistsError(
            f"story number {story_number} already has file(s) in {output_folder}: "
            f"{existing} -- remove or rename the stale file first rather than "
            f"letting two files silently coexist under the same number."
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Title: {title}\n")
        if author:
            f.write(f"Author: {author}\n")
        if date:
            f.write(f"Date: {date}\n")
        if url:
            f.write(f"URL: {url}\n")
        if source_pdf:
            f.write(f"Source PDF: {os.path.basename(source_pdf)}\n")
        f.write("=" * 50 + "\n\n")
        f.write(body_text)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one story PDF to extracted_articles_boilerplate/'s "
            "{story_number}-{slug}.txt format. Story number and title are "
            "required -- pull them from the GT Expansion List, don't guess "
            "from the PDF's own text."
        ),
        epilog="Example: python storypdf_to_text.py 176 'The Roadblocks to Relief' "
               "176.pdf --author 'Madeleine Bair' --date 'May 10, 2022'",
    )
    parser.add_argument("story_number", type=int, help="Story number, e.g. 176")
    parser.add_argument("title", help="Article title, exactly as it appears in the GT Expansion List")
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--author", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FOLDER,
                         help=f"Output folder (default: {DEFAULT_OUTPUT_FOLDER}/)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    body_text = extract_pdf_text(args.pdf_path)
    if not body_text:
        print(f"Error: no extractable text in {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(body_text)} characters from {args.pdf_path}.")
    print(
        "NOTE: this only strips the repeating browser-print header/footer "
        "lines. Site-specific chrome (nav menus, 'recommended articles', "
        "comment threads, newsletter forms, etc.) is NOT auto-stripped -- "
        "read the saved file and trim that by hand before treating it as "
        "clean corpus text."
    )

    out_path = save_article(
        story_number=args.story_number,
        title=args.title,
        body_text=body_text,
        author=args.author,
        date=args.date,
        url=args.url,
        source_pdf=args.pdf_path,
        output_folder=args.output,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
