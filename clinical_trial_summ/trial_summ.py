from transformers import pipeline
import json
import argparse

# Load the Hugging Face Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def load_clinical_trial_data(file_path):
    """Load raw clinical trial text from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_clinical_trial(trial_text):
    """Summarize a clinical trial report using a local model."""
    summary = summarizer(trial_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def process_trials(input_file, output_file):
    """Process multiple clinical trials and store structured summaries."""
    trials = load_clinical_trial_data(input_file)
    structured_summaries = []

    for trial in trials:
        summary = summarize_clinical_trial(trial["text"])
        structured_summaries.append({"trial_id": trial["id"], "summary": summary})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_summaries, f, indent=4)

    print(f"Processed {len(trials)} trials. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize Clinical Trial Reports using a local model")
    parser.add_argument("input_file", type=str, help="Path to input JSON file containing trial reports")
    parser.add_argument("output_file", type=str, help="Path to save the structured summaries")
    args = parser.parse_args()

    process_trials(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
