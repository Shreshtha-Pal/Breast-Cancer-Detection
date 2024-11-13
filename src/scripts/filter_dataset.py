import cv2
import os
import pandas as pd
import shutil
from config.paths import BIN_DIR, DATASET_LABELS_WITH_BRIGHTNESS, DATASET_LABELS_FILTERED, DATASET_LABELS_DISCARDED
from tqdm import tqdm



def filter_dataset(copy_discarded_to_bin):
    if (copy_discarded_to_bin):
        os.makedirs(f'{BIN_DIR}/0', exist_ok=True)
        os.makedirs(f'{BIN_DIR}/1', exist_ok=True)

    dataset_labels_with_brightness_df = pd.read_csv(DATASET_LABELS_WITH_BRIGHTNESS)

    dataset_labels_filtered = []
    dataset_labels_discarded = []

    print("\n")
    for _, row in tqdm(dataset_labels_with_brightness_df.iterrows(), total=len(dataset_labels_with_brightness_df), desc=f"Filtering Dataset"):
        image_path = f"{row['dir']}/{row['image']}"
        imgage = cv2.imread(image_path)

        if (
            (imgage.shape == (50, 50, 3))
            and
            (
                ((row['class'] == 0) and (165 < row['brightness'] <= 225))
                or
                ((row['class'] == 1) and (130 < row['brightness'] <= 190))
            )
        ):
            dataset_labels_filtered.append(row)

        else:
            dataset_labels_discarded.append(row)
            if (copy_discarded_to_bin):
                shutil.copy2(image_path, f"{BIN_DIR}/{row['class']}/{row['image']}")

    dataset_labels_filtered_df = pd.DataFrame(data=dataset_labels_filtered)
    dataset_labels_filtered_df.to_csv(DATASET_LABELS_FILTERED, index=False)
    print(f"\nFiltered Labels Saved In - {DATASET_LABELS_FILTERED}")

    dataset_labels_discarded_df = pd.DataFrame(data=dataset_labels_discarded)
    dataset_labels_discarded_df.to_csv(DATASET_LABELS_DISCARDED, index=False)
    print(f"\nDiscarded Labels Saved In - {DATASET_LABELS_DISCARDED}")

    counts_filtered = dataset_labels_filtered_df['class'].value_counts().reindex([0, 1], fill_value=0)
    counts_discarded = dataset_labels_discarded_df['class'].value_counts().reindex([0, 1], fill_value=0)

    summary_df = pd.DataFrame(
        {
            'class0': [counts_filtered[0], counts_discarded[0]],
            'class1': [counts_filtered[1], counts_discarded[1]]
        },
        index=['filtered', 'discarded']
    )

    print(f"\n{summary_df}")



if __name__ == "__main__":

    copy_discarded_to_bin = False

    filter_dataset(copy_discarded_to_bin)
