# System Requirements

## Required System Dependencies

### OCR Support

PDF pages with almost no extractable text (image-only or scanned pages) are OCR'd. That needs two
system packages on the server's PATH. They are not Python packages, so `pip`/`uv` won't install them:

- **Poppler** (`pdfinfo`, `pdftoppm`): renders the PDF page to an image (used by `pdf2image`)
- **Tesseract OCR** with English data: reads the text from that image (used by `pytesseract`)

If either is missing, a PDF upload that needs OCR fails with **503**. Text-only PDFs and other file
types are not affected. Production installs both through `deployment/ansible.yml`.

**macOS:**

```bash
brew install poppler tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng
```

### NLP Support

**spaCy English Model** - Required for text processing

```bash
python -m spacy download en_core_web_sm
```

## Verification

Check installations:

```bash
# Verify Poppler and Tesseract (English must be listed)
pdfinfo -v
pdftoppm -v
tesseract --version
tesseract --list-langs

# Verify spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded')"
```

At startup the app logs `OCR dependencies found on PATH: ...`, or an ERROR naming the missing binaries.
