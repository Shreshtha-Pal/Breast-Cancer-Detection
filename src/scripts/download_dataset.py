import os
import subprocess
import zipfile
from config.paths import DOWNLOAD_DIR
from tqdm import tqdm



def unzip_dataset(dataset_identifier):

    zip_name = dataset_identifier.split('/')[-1]

    zip_path = f'{DOWNLOAD_DIR}/{zip_name}.zip'
    extract_to = f'{DOWNLOAD_DIR}/{zip_name}'

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        file_list = zip_ref.namelist()

        print("\n")
        for file in tqdm(file_list, total=len(file_list), desc=f"Extracting Dataset To - {extract_to}"):
            zip_ref.extract(file, extract_to)



def download_dataset(dataset_identifier):

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    download_command = f'kaggle datasets download -d {dataset_identifier} -p {DOWNLOAD_DIR}'

    try:
        print("\n")
        subprocess.run(download_command, check=True, shell=True)

    except Exception as e:
        print(f'\nError Downloading Dataset.\n\n{e}')
        return



if __name__ == "__main__":

    os.environ['KAGGLE_USERNAME'] = ''
    os.environ['KAGGLE_KEY'] = ''

    dataset_identifier = 'paultimothymooney/breast-histopathology-images'

    download_dataset(dataset_identifier)

    unzip_dataset(dataset_identifier)
