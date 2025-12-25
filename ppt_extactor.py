import io
import json
import os
import sys

# specific imports
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from PIL import Image
    import pytesseract
except ImportError as e:
    print("Error: Missing required libraries.")
    print(f"Details: {e}")
    print("Please run: pip install python-pptx pytesseract Pillow")
    sys.exit(1)

# --- CONFIGURATION ---
# Explicitly set the path to the Tesseract executable
# If your path is different, change it here.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_shape_text(shape):
    """
    Recursively extracts text from a shape (handles TextBoxes, Tables, and Groups).
    Returns a list of strings.
    """
    text_parts = []

    # 1. Text Frames (Standard TextBoxes, Shapes)
    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            # Join runs to get clean paragraph text
            p_text = "".join(run.text for run in paragraph.runs)
            if p_text.strip():
                text_parts.append(p_text)

    # 2. Tables
    if shape.has_table:
        for row in shape.table.rows:
            row_cells = []
            for cell in row.cells:
                if cell.text_frame and cell.text.strip():
                    row_cells.append(cell.text.strip())
            if row_cells:
                text_parts.append(" | ".join(row_cells))

    # 3. Groups (Recursive extraction)
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sub_shape in shape.shapes:
            text_parts.extend(get_shape_text(sub_shape))

    return text_parts

def get_shape_images(shape):
    """
    Recursively extracts image blobs from a shape (handles Pictures, Placeholders, and Groups).
    Returns a list of binary image blobs.
    """
    images = []

    # 1. Picture / Placeholder with image
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or shape.shape_type == MSO_SHAPE_TYPE.PLACEHOLDER:
        if hasattr(shape, "image"):
            # We use the blob (bytes) directly
            images.append(shape.image.blob)

    # 2. Groups (Recursive extraction)
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sub_shape in shape.shapes:
            images.extend(get_shape_images(sub_shape))

    return images

def process_pptx(file_path):
    """
    Main processing function.
    Iterates through slides, extracts text and OCRs images.
    Returns a list of dictionaries.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    print(f"Loading presentation: {file_path}...")
    try:
        prs = Presentation(file_path)
    except Exception as e:
        print(f"Error opening presentation: {e}")
        return []

    output_data = []

    print(f"Found {len(prs.slides)} slides. Processing...")

    for i, slide in enumerate(prs.slides):
        slide_num = i + 1
        
        # Containers for this specific slide
        slide_text_content = []
        slide_image_ocr_content = []

        # Iterate over all shapes in the slide
        for shape in slide.shapes:
            
            # --- Extract Visible Text ---
            texts = get_shape_text(shape)
            slide_text_content.extend(texts)

            # --- Extract Images and Run OCR ---
            image_blobs = get_shape_images(shape)
            
            for blob in image_blobs:
                try:
                    # Convert bytes to PIL Image
                    image = Image.open(io.BytesIO(blob))
                    
                    # Perform OCR using Tesseract
                    # --psm 6 assumes a single uniform block of text (good for slide images)
                    text = pytesseract.image_to_string(image, config='--psm 6')
                    
                    if text.strip():
                        slide_image_ocr_content.append(text.strip())
                except Exception as e:
                    # Silently ignore OCR errors for cleaner output
                    pass

        # Format the data exactly as requested
        slide_entry = {
            "slide": slide_num,
            "text": "\n".join(slide_text_content),
            "image_text": "\n".join(slide_image_ocr_content)
        }

        output_data.append(slide_entry)
        print(f"Processed Slide {slide_num}")

    return output_data

if __name__ == "__main__":
    # --- USER INPUT ---
    # Update this filename to match your local file
    INPUT_FILE = "input_presentation.pptx"

    # Process
    results = process_pptx(INPUT_FILE)

    # --- MANDATORY OUTPUT FORMAT ---
    # Printing the raw JSON string to stdout
    print("\n--- FINAL JSON OUTPUT ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
