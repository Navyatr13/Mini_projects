import argparse
from collections import defaultdict
from datetime import datetime

def parse_log_file(filename, start_date, end_date):
    events_count = defaultdict(int)
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(" ", 3)
            if len(parts) < 4:
                continue

            time_stamp, category, desc = parts[0] + " "+ parts[1], parts[2], parts[3]
            print(time_stamp, category)
            try:
                time_stamp = datetime.strptime( time_stamp, "%Y-%m-%d %H:%M:%S")
                if time_stamp < start_date:
                    continue
                if time_stamp > end_date:
                    continue
                events_count[category] += 1
            except:
                continue
    return events_count

def display(events_counts):
    sorted_events = sorted(events_counts.items(), key = lambda x:x[1], reverse = True)
    print(sorted_events)

    for event, count in sorted_events:
        print(f"{event}: {count} occurrences")
    if events_counts:
        most_frequent = max(events_counts, key=events_counts.get)
        print(f"\nMost Frequent Event: {most_frequent} ({events_counts[most_frequent]} times)")



def main():
    parser = argparse.ArgumentParser(description= "Provide a log file")
    parser.add_argument("file_name", type=str, help = "Path to log file", default=None)
    parser.add_argument("--start_date", type= str, help = "Start date (YYYY-MM-DD)" , default= None)
    parser.add_argument("--end_date", type=str, help = "End date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    events_counts = parse_log_file(args.file_name, start_date, end_date)
    print(events_counts)
    display(events_counts)


if __name__ == "__main__":
    main()


