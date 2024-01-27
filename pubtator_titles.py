import csv


def extract_titles_and_ids(file_path):
    titles_and_ids = []

    with open(file_path, 'r') as file:
        for line in file:
            if '|t|' in line:
                parts = line.split('|')
                idx = parts[0].strip()  # Extract the index
                title = parts[2].strip()  # Extract the title
                titles_and_ids.append((idx, title))

    return titles_and_ids


def write_to_csv(titles_and_ids, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Idx', 'Title'])  # Header
        for idx, title in titles_and_ids:
            writer.writerow([idx, title])


# Replace 'your_file.txt' with the path to your text file
file_path = 'filtered_pubtator_responses_V5.txt'

# Specify the path for the new CSV file
csv_file_path = 'V5_title.csv'

titles_and_ids = extract_titles_and_ids(file_path)
write_to_csv(titles_and_ids, csv_file_path)

print("Titles and indices have been written to", csv_file_path)
