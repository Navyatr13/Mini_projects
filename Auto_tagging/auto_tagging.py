import json
import pandas as pd
from transformers import pipeline

# Load pre-trained text classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def load_clinical_trials(json_file):
    """Load clinical trial summaries from a JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def classify_trial(trial, labels):
    """Classify clinical trial into categories using LLM-based classification."""
    result = classifier(trial["summary"], candidate_labels=labels)
    return result["labels"][0]  # Most probable category


def process_trials(json_file, output_csv):
    """Auto-tag clinical trials and save the results."""
    trials = load_clinical_trials(json_file)
    labels = ["Oncology", "Cardiology", "Neurology", "Infectious Disease", "Diabetes"]

    processed_data = []
    for trial in trials:
        category = classify_trial(trial, labels)
        trial["category"] = category
        processed_data.append(trial)

    df = pd.DataFrame(processed_data)
    df.to_csv(output_csv, index=False)
    print(f"Auto-tagged clinical trials saved to {output_csv}")


def main():
    json_file = "structured_clinical_trials.json"
    output_csv = "auto_tagged_clinical_trials.csv"
    process_trials(json_file, output_csv)
    print("Clinical trial auto-tagging completed.")


if __name__ == "__main__":
    main()
