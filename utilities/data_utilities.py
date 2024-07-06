import os
import sys
import zipfile

import kaggle



def download_and_extract_kaggle_data(competition_name, data_dir = 'data'):
    zip_file_name = f"{competition_name}.zip"
    
    # Checking if the data directory already exists and contains files
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"Data for {competition_name} already exists in '{data_dir}/' directory.")
        return
    
    # Creating data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok = True)
    
    # Checking if the zip file already exists
    if not os.path.exists(zip_file_name):
        print(f"Downloading data for {competition_name}...")
        kaggle.api.competition_download_files(competition_name)
    else:
        print(f"Zip file for {competition_name} already exists.")
    
    # Extracting the contents of the zip file to the data directory
    print(f"Extracting files to '{data_dir}/' directory...")
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Removing the zip file after extraction
    os.remove(zip_file_name)
    
    print(f"Files extracted to '{data_dir}/' directory")