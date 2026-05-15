"""
Document Processor
------------------
Handles messy legal documents: scanned PDFs, handwritten notes,
low-resolution images, partially illegible records.
Extracts clean text + structured fields for downstream use.
"""

import os
import re
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import io


@dataclass
class ProcessedDocument:
    """Output of document processing stage."""
    source_path: str
    raw_text: str
    cleaned_text: str
    structured_fields: dict = field(default_factory=dict)
    pages: list = field(default_factory=list)
    extraction_method: str = "unknown"
    confidence: float = 0.0
    warnings: list = field(default_factory=list)


class DocumentProcessor:
    """
    Processes messy legal documents into clean, usable text.
    
    Strategy:
    1. Try pdfplumber first (best for digital PDFs)
    2. Fall back to PyMuPDF
    3. Fall back to OCR for scanned/image pages
    4. Clean and normalize the extracted text
    5. Extract structured fields (dates, parties, case numbers etc.)
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Patterns for legal document structured fields
        self.field_patterns = {
            "case_number": r"(?i)case\s*(?:no|number|#)[\.\:\s]*([A-Z0-9\-\/]+)",
            "date": r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b",
            "parties": r"(?i)(?:between|plaintiff|defendant|claimant|respondent)[\:\s]+([^\n]{5,80})",
            "court": r"(?i)(?:in the|before the)\s+([\w\s]+court[\w\s]*)",
            "subject": r"(?i)(?:re|subject|matter|regarding)[\:\s]+([^\n]{5,100})",
            "amount": r"(?i)(?:amount|sum|total|value)[\:\s]*(?:of\s*)?(?:USD|INR|GBP|€|\$|£)?\s*([\d,]+(?:\.\d{2})?)",
        }

    def process(self, file_path: str) -> ProcessedDocument:
        """Main entry point. Processes any document file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self._process_image(file_path)
        elif ext == ".txt":
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_pdf(self, file_path: str) -> ProcessedDocument:
        """Process PDF — tries digital extraction first, falls back to OCR."""
        pages_text = []
        warnings = []
        extraction_method = "pdfplumber"
        total_confidence = 0.0

        try:
            # Strategy 1: pdfplumber for digital PDFs
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        pages_text.append({
                            "page": i + 1,
                            "text": text,
                            "method": "pdfplumber"
                        })
                        total_confidence += 0.9
                    else:
                        # Page is likely scanned — use OCR
                        warnings.append(f"Page {i+1}: low text yield, using OCR")
                        ocr_result = self._ocr_pdf_page(file_path, i)
                        pages_text.append({
                            "page": i + 1,
                            "text": ocr_result["text"],
                            "method": "ocr"
                        })
                        total_confidence += ocr_result["confidence"]
                        extraction_method = "mixed"

        except Exception as e:
            warnings.append(f"pdfplumber failed: {e}, trying PyMuPDF")
            # Strategy 2: PyMuPDF fallback
            try:
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        pages_text.append({"page": i+1, "text": text, "method": "pymupdf"})
                        total_confidence += 0.8
                    else:
                        ocr_result = self._ocr_pdf_page(file_path, i)
                        pages_text.append({"page": i+1, "text": ocr_result["text"], "method": "ocr"})
                        total_confidence += ocr_result["confidence"]
                extraction_method = "pymupdf"
            except Exception as e2:
                warnings.append(f"PyMuPDF also failed: {e2}")

        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(
            p["text"] for p in pages_text if p["text"]
        )
        cleaned_text = self._clean_text(raw_text)
        avg_confidence = total_confidence / max(len(pages_text), 1)

        return ProcessedDocument(
            source_path=file_path,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            structured_fields=self._extract_structured_fields(cleaned_text),
            pages=pages_text,
            extraction_method=extraction_method,
            confidence=round(avg_confidence, 2),
            warnings=warnings
        )

    def _ocr_pdf_page(self, file_path: str, page_index: int) -> dict:
        """Run OCR on a specific PDF page."""
        try:
            doc = fitz.open(file_path)
            page = doc[page_index]
            
            # Render at high resolution for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            image = Image.open(io.BytesIO(img_data))
            image = self._preprocess_image_for_ocr(image)
            
            # OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            # Calculate average confidence
            confidences = [int(c) for c in ocr_data['conf'] if c != '-1']
            avg_conf = sum(confidences) / max(len(confidences), 1) / 100
            
            return {"text": text, "confidence": avg_conf}
            
        except Exception as e:
            return {"text": f"[OCR failed for page {page_index+1}: {e}]", "confidence": 0.0}

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        Handles low-res scans and noisy documents.
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (OCR works better at 300+ DPI equivalent)
        w, h = image.size
        if w < 1000:
            scale = 1000 / w
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        return image

    def _process_image(self, file_path: str) -> ProcessedDocument:
        """Process image file directly with OCR."""
        image = Image.open(file_path)
        image = self._preprocess_image_for_ocr(image)
        
        text = pytesseract.image_to_string(image, config='--psm 6')
        cleaned = self._clean_text(text)
        
        return ProcessedDocument(
            source_path=file_path,
            raw_text=text,
            cleaned_text=cleaned,
            structured_fields=self._extract_structured_fields(cleaned),
            pages=[{"page": 1, "text": text, "method": "ocr"}],
            extraction_method="ocr",
            confidence=0.75,
            warnings=[]
        )

    def _process_text(self, file_path: str) -> ProcessedDocument:
        """Process plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        cleaned = self._clean_text(text)
        return ProcessedDocument(
            source_path=file_path,
            raw_text=text,
            cleaned_text=cleaned,
            structured_fields=self._extract_structured_fields(cleaned),
            pages=[{"page": 1, "text": text, "method": "text"}],
            extraction_method="text",
            confidence=1.0,
            warnings=[]
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        Handles OCR artifacts, inconsistent formatting, etc.
        """
        if not text:
            return ""
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR errors in legal documents
        ocr_fixes = {
            r'\bl\b(?=\d)': '1',      # lowercase L before digits → 1
            r'(?<=\d)O(?=\d)': '0',   # O between digits → 0
            r'§': 'Section',           # Section symbol
            r'©': '(c)',
        }
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove page numbers standing alone
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Strip leading/trailing whitespace per line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines).strip()
        
        return text

    def _extract_structured_fields(self, text: str) -> dict:
        """Extract key legal fields from text using regex patterns."""
        fields = {}
        
        for field_name, pattern in self.field_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Deduplicate and take top 3
                unique_matches = list(dict.fromkeys(
                    m.strip() for m in matches if m.strip()
                ))[:3]
                fields[field_name] = unique_matches
        
        return fields