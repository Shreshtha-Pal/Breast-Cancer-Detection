import os
import pandas as pd
from config.paths import DATASET_DIR, DATASET_LABELS, LABELS_DIR



def label_dataset():

    os.makedirs(LABELS_DIR, exist_ok=True)

    dataset_labels = []

    for (dirpath, _, filenames) in os.walk(DATASET_DIR):

        for filename in filenames:

            if 'class0' in filename:
                idc_class = 0
            elif 'class1' in filename:
                idc_class = 1
            else:
                continue

            dataset_labels.append({'dir':dirpath, 'image':filename, 'class':idc_class})

    labels_df = pd.DataFrame(data=dataset_labels)
    labels_df.to_csv(DATASET_LABELS, index=False)

    print(f'\nLabels Saved In - {DATASET_LABELS}')



if __name__ == "__main__":

    label_dataset()
