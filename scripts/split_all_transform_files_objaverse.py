import os
import sys
import subprocess


def process_directory(full_path):
    command = f"python scripts/split_transform_file_objaverse.py {full_path}/"
    subprocess.run(command, shell=True)


def main(base_directory):
    for dir_name in os.listdir(base_directory):
        full_path = os.path.join(base_directory, dir_name)
        if os.path.isdir(full_path):
            process_directory(full_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <base_directory>")
    else:
        base_directory = sys.argv[1]
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        else:
            main(base_directory)
