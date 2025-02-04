import json
import pytesseract
from pdf2image import convert_from_path
import re
from transformers import pipeline

# Initialize LLM-based NLP pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extract_text_from_pdf(pdf_path):
    """Extract text from a clinical trial PDF using OCR."""
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text


def extract_structured_data(text):
    """Extract key fields from raw text using regex."""
    trial_data = {}
    trial_data["trial_id"] = re.search(r'Trial ID:\s*(NCT\d+)', text).group(1) if re.search(r'Trial ID:\s*(NCT\d+)',
                                                                                            text) else "Unknown"
    trial_data["title"] = re.search(r'Title:\s*(.+)', text).group(1) if re.search(r'Title:\s*(.+)', text) else "Unknown"
    trial_data["study_population"] = re.search(r'Study Population:\s*(.+)', text).group(1) if re.search(
        r'Study Population:\s*(.+)', text) else "Unknown"
    trial_data["intervention"] = re.search(r'Intervention:\s*(.+)', text).group(1) if re.search(r'Intervention:\s*(.+)',
                                                                                                text) else "Unknown"
    trial_data["results"] = re.search(r'Results:\s*(.+)', text).group(1) if re.search(r'Results:\s*(.+)',
                                                                                      text) else "Unknown"
    trial_data["funding"] = re.search(r'Funding:\s*(.+)', text).group(1) if re.search(r'Funding:\s*(.+)',
                                                                                      text) else "Unknown"

    return trial_data


def summarize_text(text):
    """Summarize extracted clinical trial data using an LLM."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def process_pdf(pdf_path, output_json):
    raw_text = extract_text_from_pdf(pdf_path)
    structured_data = extract_structured_data(raw_text)
    structured_data["summary"] = summarize_text(raw_text)

    with open(output_json, "w") as f:
        json.dump(structured_data, f, indent=4)
    print(f"Structured trial data saved to {output_json}")


def main():
    pdf_path = "sample_doc.pdf"
    output_json = "structured_clinical_trial.json"
    process_pdf(pdf_path, output_json)
    print("Clinical trial data extraction completed.")


if __name__ == "__main__":
    main()
