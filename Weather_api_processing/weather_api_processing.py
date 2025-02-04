import requests
import argparse
import json
import os


def fetch_weather(city):
    API_KEY = "b54d9d74f0e2410dbc3b879e7a36a4d4"  # Replace with your OpenWeatherMap API key
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    params = {
        "q" : city,
        "units" : "metric",
        "appid" : API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx and 5xx)
        return response.json()
    except   requests.exceptions.RequestException as e:
        print(f"Error: Unable to fetch data for {city}. Reason: {e}")
        return None

def process_weather_data(data):
    if not data:
        return None
    try:
        weather_info = {
            "city" : data.get("name"),
            "Temperature" : data["main"].get("temp"),
            "Humidity" : data['main'].get("humidity"),
            "Weather Condition" : data["weather"][0].get("description") if "weather" in data and data["weather"] else "N/A"
        }
        return weather_info
    except KeyError as e:
        print(f"Error processing weather data: Missing key {e}")
        return None


def save_to_json(data, file_name):
    try:
        # Check if file exists and load existing data
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):  # Ensure it's a list
                        existing_data = [existing_data]
                except json.JSONDecodeError:  # Handle empty or invalid JSON
                    existing_data = []
        else:
            existing_data = []

        # Append new data
        existing_data.append(data)

        # Write updated list back to file
        with open(file_name, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Data appended to {file_name}")

    except IOError as e:
        print(f"Error saving data: {e}")

def main():
    parser = argparse.ArgumentParser(description= "Weather api")
    parser.add_argument("cities", nargs="+", help= "List of cities", default= "San Fransisco")

    args = parser.parse_args()

    for city in args.cities:
        raw_data = fetch_weather(city)
        processed_data = process_weather_data(raw_data)
        print(processed_data)
        save_to_json(processed_data, file_name="result_file.json")
if __name__ == "__main__":
    main()


