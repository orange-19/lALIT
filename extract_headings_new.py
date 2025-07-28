import pdfplumber
import json
import os
import glob
import argparse
from pathlib import Path

def extract_headings_from_pdf(pdf_path):
    """
    Extract headings from all pages of a PDF.
    Focuses on actual headings based on font size and formatting, not just bold text.
    """
    all_headings = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return []
            
            for page_number, page in enumerate(pdf.pages, start=1):
                lines = page.extract_text_lines(extra_attrs=["size", "fontname"])
                
                if not lines:
                    continue
                
                # Get font sizes to determine what constitutes a heading
                font_sizes = []
                for line in lines:
                    font_size = line.get('size', 12.0)
                    text = line.get('text', '').strip()
                    if text and len(text) > 2:  # Only consider substantial text
                        font_sizes.append(font_size)
                
                if not font_sizes:
                    continue
                
                # Calculate font size thresholds for headings
                avg_font_size = sum(font_sizes) / len(font_sizes)
                max_font_size = max(font_sizes)
                
                # Consider text as heading if font size is significantly larger than average
                heading_threshold = avg_font_size + (max_font_size - avg_font_size) * 0.3
                
                for line_num, line in enumerate(lines, 1):
                    font_size = line.get('size', 12.0)
                    font_name = line.get('fontname', '')
                    text = line.get('text', '').strip()
                    top_position = line.get('top', 0)
                    
                    # Skip empty or very short text
                    if not text or len(text) < 3:
                        continue
                    
                    # Skip text that's just numbers or symbols
                    if text.replace('.', '').replace(')', '').replace('(', '').replace('-', '').isdigit():
                        continue
                    
                    # Skip text that's mostly punctuation
                    if len([c for c in text if c in '.,;:()[]{}']) > len(text) * 0.5:
                        continue
                    
                    # Determine if this is a heading based on font size
                    is_heading = font_size >= heading_threshold
                    
                    # Additional checks for heading characteristics
                    if is_heading:
                        # Determine heading level based on font size
                        if font_size >= max_font_size * 0.9:
                            level = "H1"
                        elif font_size >= max_font_size * 0.7:
                            level = "H2"
                        else:
                            level = "H3"
                        
                        heading_info = {
                            "level": level,
                            "text": text,
                            "page": page_number
                        }
                        all_headings.append(heading_info)
            
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return []
    
    return all_headings

def extract_headings_from_first_page(pdf_path):
    """
    Extract headings from the first page of a PDF only.
    """
    headings = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return []
            
            # Process only the first page
            first_page = pdf.pages[0]
            lines = first_page.extract_text_lines(extra_attrs=["size", "fontname"])
            
            if not lines:
                return []
            
            # Get font sizes to determine what constitutes a heading
            font_sizes = []
            for line in lines:
                font_size = line.get('size', 12.0)
                text = line.get('text', '').strip()
                if text and len(text) > 2:  # Only consider substantial text
                    font_sizes.append(font_size)
            
            if not font_sizes:
                return []
            
            # Calculate font size thresholds for headings
            avg_font_size = sum(font_sizes) / len(font_sizes)
            max_font_size = max(font_sizes)
            
            # Consider text as heading if font size is significantly larger than average
            heading_threshold = avg_font_size + (max_font_size - avg_font_size) * 0.3
            
            for line_num, line in enumerate(lines, 1):
                font_size = line.get('size', 12.0)
                text = line.get('text', '').strip()
                
                # Skip empty or very short text
                if not text or len(text) < 3:
                    continue
                
                # Skip text that's just numbers or symbols
                if text.replace('.', '').replace(')', '').replace('(', '').replace('-', '').isdigit():
                    continue
                
                # Skip text that's mostly punctuation
                if len([c for c in text if c in '.,;:()[]{}']) > len(text) * 0.5:
                    continue
                
                # Determine if this is a heading based on font size
                is_heading = font_size >= heading_threshold
                
                # Additional checks for heading characteristics
                if is_heading:
                    headings.append(text)
            
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return []
    
    return headings

def process_pdf_files(directory_path):
    """
    Process all PDF files in the directory and extract headings.
    Store each PDF's output separately.
    """
    # Find all PDF files in the directory
    pdf_pattern = os.path.join(directory_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    for pdf_file in pdf_files:
        print(f"Processing: {os.path.basename(pdf_file)}")
        
        # Extract first page headings for title
        first_page_headings = extract_headings_from_first_page(pdf_file)
        title = "  ".join(first_page_headings) if first_page_headings else "Untitled"
        
        # Extract all headings for outline
        all_headings = extract_headings_from_pdf(pdf_file)
        
        if all_headings:
            # Create output structure
            output = {
                "title": title,
                "outline": all_headings
            }
            
            # Save to individual JSON file
            filename = os.path.basename(pdf_file)
            output_file = os.path.join(directory_path, f"{Path(filename).stem}_output.json")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"  - Saved {len(all_headings)} headings to {os.path.basename(output_file)}")
            except Exception as e:
                print(f"  - Error saving {output_file}: {str(e)}")
        else:
            print(f"  - No headings found in {os.path.basename(pdf_file)}")

def main():
    parser = argparse.ArgumentParser(description="Extract headings from PDF files and save as separate JSON files")
    parser.add_argument("--pdf", type=str, help="Specific PDF file to process")
    parser.add_argument("--directory", type=str, help="Directory to search for PDF files")
    args = parser.parse_args()
    
    # Determine the directory to process
    if args.directory:
        directory = args.directory
    elif args.pdf and os.path.exists(args.pdf):
        # If specific PDF is provided, process just that one
        directory = os.path.dirname(os.path.abspath(args.pdf))
        pdf_files = [args.pdf]
    else:
        # Use current directory
        directory = os.getcwd()
    
    print(f"Processing PDFs in directory: {directory}")
    
    # Process PDFs
    if args.pdf and os.path.exists(args.pdf):
        # Process specific PDF
        print(f"Processing specific PDF: {args.pdf}")
        
        # Extract first page headings for title
        first_page_headings = extract_headings_from_first_page(args.pdf)
        title = "  ".join(first_page_headings) if first_page_headings else "Untitled"
        
        # Extract all headings for outline
        all_headings = extract_headings_from_pdf(args.pdf)
        
        if all_headings:
            output = {
                "title": title,
                "outline": all_headings
            }
            
            filename = os.path.basename(args.pdf)
            output_file = os.path.join(directory, f"{Path(filename).stem}_output.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(all_headings)} headings to {os.path.basename(output_file)}")
        else:
            print("No headings found")
    else:
        # Process all PDFs in directory
        process_pdf_files(directory)
    
    return 0

if __name__ == "__main__":
    main()
