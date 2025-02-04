import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://clinicaltrials.gov/search?cond={}&page={}"  # Modify as per actual site structure
HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_trials(condition, max_pages=3):
    """Simple scraper for clinical trials with trial IDs."""
    trials = []
    for page in range(1, max_pages + 1):
        url = BASE_URL.format(condition.replace(" ", "+"), page)
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        trial_list = soup.find_all("div", class_="ct-card")  # Adjust selector if needed
        print()
        for trial in trial_list:
            trial_id_tag = trial.find("div", class_="nct-id")
            trial_id = trial_id_tag.text.strip() if trial_id_tag else "N/A"

            title_tag = trial.find("h2")
            title = title_tag.text.strip() if title_tag else "N/A"

            summary_tag = trial.find("p")
            summary = summary_tag.text.strip() if summary_tag else "N/A"

            link_tag = trial.find("a")
            link = "https://clinicaltrials.gov" + link_tag["href"] if link_tag else "N/A"

            trials.append({
                "trial_id": trial_id,
                "title": title,
                "summary": summary,
                "link": link
            })

        time.sleep(1)  # Be respectful to the server

    return trials


def save_to_csv(trials, filename="clinical_trials.csv"):
    """Save scraped data to a CSV file."""
    df = pd.DataFrame(trials)
    df.to_csv(filename, index=False)
    print(f"Saved {len(trials)} clinical trials to {filename}")


def main():
    """Main function to execute the scraper."""
    condition = "diabetes"
    trials = scrape_trials(condition)
    if trials:
        save_to_csv(trials)
    else:
        print("No trials found.")


if __name__ == "__main__":
    main()
