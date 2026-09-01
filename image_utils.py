import cv2
import numpy as np
import easyocr

# Initialize OCR reader (cached for performance)
_ocr_reader = None

def _get_ocr_reader():
    """Get or initialize the OCR reader (lazy loading for better performance)."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def extract_text_from_image(image):
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Extracted text as string
    """
    if image is None:
        return ""
    
    try:
        # Get OCR reader
        reader = _get_ocr_reader()
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR
        results = reader.readtext(binary)
        
        # Extract and combine text
        if results:
            text = "\n".join([text_data[1] for text_data in results])
            return text.strip()
        return ""
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""


def read_image(uploaded_file):
    """Read an uploaded image file into a NumPy OpenCV image."""
    if uploaded_file is None:
        return None

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    return image


def preprocess_image(image):
    """Convert image to grayscale and reduce noise for downstream processing."""
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def enhance_contrast(image):
    """Apply CLAHE to improve visibility in medical images."""
    if image is None:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def detect_edges(image):
    """Compute Canny edges for image feature detection."""
    if image is None:
        return None

    return cv2.Canny(image, 50, 150)
