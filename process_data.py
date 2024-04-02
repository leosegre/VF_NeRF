import os
import sys
import subprocess


def process_directory(dir_name, base_directory, output_directory):
    command = f"ns-process-data images --data {base_directory}/{dir_name}/images/ --output-dir {output_directory}/{dir_name}"
    subprocess.run(command, shell=True)


def main(base_directory, output_directory):
    for dir_name in os.listdir(base_directory):
        full_path = os.path.join(base_directory, dir_name)
        if os.path.isdir(full_path):
            process_directory(dir_name, base_directory, output_directory)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <base_directory> <output_directory>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            main(base_directory, output_directory)
