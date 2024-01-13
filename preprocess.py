import json
import csv

# Read JSON data from file
with open('dataset.json', 'r') as json_file:
    data = json.load(json_file)

# Specify the output CSV file name
output_csv_file = 'dataset.csv'

# Define field names for CSV
fieldnames = ['gmap_id', 'user', 'rating', 'timestamp', 'location_name', 'user_name']

# Write JSON data to a CSV file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write headers
    writer.writeheader()
    
    # Write rows
    for entry in data:
        writer.writerow(entry)
