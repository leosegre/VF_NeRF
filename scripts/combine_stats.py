import os
import json
import sys
import csv

def combine_json_files(directory_path):
    # Initialize an empty dictionary to store combined data
    combined_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # Read JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

                for key, value in data.items():
                    # # Check if "translation_rmse" key exists in the nested dictionary
                    # if "translation_rmse" in value:
                    #     # Add "translation_rmse_square" key with the squared value
                    #     value["translation_rmse_square"] = value["translation_rmse"] * 100

                    # Combine the data into the overall dictionary
                    combined_data[key] = value
    # Generate the output file path in the same directory with the name "combined_stats.json"

    # Sort the combined data by scene name
    combined_data = {k: v for k, v in sorted(combined_data.items())}

    # Generate the output file path in the same directory with the name "combined_stats.csv"
    output_file_path = os.path.join(directory_path, "combined_stats.csv")

    # Write the combined data to a new CSV file
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Scene Name", "Translation RMSE x 100", "Rotation RMSE", "t0 Translation RMSE x 100", "t0 Rotation RMSE"])
        for scene, stats in combined_data.items():
            if "t0_translation_rmse_100" in stats["best"]:
                writer.writerow([scene, stats["best"]["translation_rmse_100"], stats["best"]["rotation_rmse"], stats["best"]["t0_translation_rmse_100"], stats["best"]["t0_rotation_rmse"]])
            else:
                writer.writerow([scene, stats["best"]["translation_rmse_100"], stats["best"]["rotation_rmse"]])
    output_file_path = os.path.join(directory_path, "combined_stats.json")

    # Write the combined data to a new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=2)

    return output_file_path

if __name__ == "__main__":
    # Check if a directory path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    # Get the directory path from the command-line argument
    directory_path = sys.argv[1]

    output_file_path = combine_json_files(directory_path)
    print(f"Combined JSON data written to {output_file_path}")