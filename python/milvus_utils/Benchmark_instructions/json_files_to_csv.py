import json
import csv
import sys
import glob
import os

def flatten_json(json_obj, parent_key='', sep='_'):
    """Flattens a nested JSON object into a single dictionary."""
    items = []
    for k, v in json_obj.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, new_key + sep + str(i), sep=sep).items())
                else:
                    items.append((new_key + sep + str(i), item))
        else:
            items.append((new_key, v))
    return dict(items)

def process_json_file(json_file_path, all_keys, all_data):
    """Processes a single JSON file, flattens its content, and updates keys and data."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        for record in data:
            flattened_record = flatten_json(record)
            all_keys.update(flattened_record.keys())
            all_data.append(flattened_record)

        return True
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file_path}' not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred while processing '{json_file_path}': {e}")
        return False

def json_to_csv_flexible_multiple(input_files_pattern, output_csv_file):
    """
    Converts multiple JSON files (matching the input pattern) to a single CSV file.

    Args:
        input_files_pattern (str): A string that can include wildcards (e.g., '*.json', 'data_*.json').
        output_csv_file (str): Path to the output CSV file.
    """
    all_keys = set()
    all_data = []
    processed_count = 0

    for input_file in glob.glob(input_files_pattern):
        if process_json_file(input_file, all_keys, all_data):
            processed_count += 1

    if not all_data:
        print("No data processed. CSV file will be empty.")
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
        return

    fieldnames = sorted(list(all_keys))

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_data:
            writer.writerow(record)

    print(f"Successfully processed {processed_count} file(s) and converted to '{output_csv_file}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_files_pattern> <output_csv_file>")
        print("  <input_files_pattern>: A string with wildcard support to match multiple JSON files (e.g., '*.json', 'data_*.json', 'dir/*.json').")
        print("  <output_csv_file>: The name of the output CSV file.")
        sys.exit(1)

    input_pattern = sys.argv[1]
    output_file = sys.argv[2]
    json_to_csv_flexible_multiple(input_pattern, output_file)

