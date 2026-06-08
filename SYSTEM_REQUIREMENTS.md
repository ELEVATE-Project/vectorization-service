# System Requirements

## Required System Dependencies

### OCR Support

**Tesseract OCR** - Required for scanned PDF text extraction

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

### NLP Support

**spaCy English Model** - Required for text processing

```bash
python -m spacy download en_core_web_sm
```

## Verification

Check installations:

```bash
# Verify Tesseract
tesseract --version

# Verify spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded')"
```
