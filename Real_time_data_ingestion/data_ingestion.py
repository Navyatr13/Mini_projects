import requests
import json
from bs4 import BeautifulSoup
from transformers import pipeline
import schedule
import time

# Initialize LLM-based summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def fetch_latest_trials():
    """Scrape the latest clinical trials from ClinicalTrials.gov."""
    url = "https://clinicaltrials.gov/ct2/results/rss.xml?cond=Cancer"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")

    trials = []
    for item in soup.find_all("item"):
        trial = {
            "title": item.title.text,
            "link": item.link.text,
            "description": item.description.text
        }
        trials.append(trial)
    return trials


def summarize_trials(trials):
    """Summarize the clinical trial descriptions using an LLM."""
    for trial in trials:
        trial["summary"] = summarizer(trial["description"], max_length=100, min_length=30, do_sample=False)[0][
            'summary_text']
    return trials


def save_trials(trials, output_file="latest_clinical_trials.json"):
    """Save structured clinical trial data to JSON."""
    with open(output_file, "w") as f:
        json.dump(trials, f, indent=4)
    print(f"Latest clinical trials saved to {output_file}")


def update_trials():
    """Fetch, summarize, and save clinical trial data in real-time."""
    trials = fetch_latest_trials()
    summarized_trials = summarize_trials(trials)
    save_trials(summarized_trials)


def main():
    # Schedule the pipeline to run every 12 hours
    schedule.every(12).hours.do(update_trials)
    print("Real-time clinical trial ingestion pipeline started...")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
