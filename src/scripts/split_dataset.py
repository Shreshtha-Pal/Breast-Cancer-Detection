import os
import pandas as pd
import shutil
from config.paths import DATASET_LABELS_FILTERED, TEST_IMAGES_DIR, TEST_LABELS, TRAIN_IMAGES_DIR, TRAIN_LABELS, VAL_IMAGES_DIR, VAL_LABELS
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def copy_images(df, target_dir):
    os.makedirs(f"{target_dir}/0", exist_ok=True)
    os.makedirs(f"{target_dir}/1", exist_ok=True)

    print("\n")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying Images To - {target_dir}"):
        image_path = f"{row['dir']}/{row['image']}"
        shutil.copy2(image_path, f"{target_dir}/{row['class']}/{row['image']}")



def split(val_size, test_size):
    labels_df = pd.read_csv(DATASET_LABELS_FILTERED)

    train_df, temp_df = train_test_split(
        labels_df,
        test_size=((val_size + test_size) / 100),
        stratify=labels_df['class'],
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (val_size + test_size)),
        stratify=temp_df['class'],
        shuffle=True
    )

    train_df.to_csv(TRAIN_LABELS, index=False)
    print(f"\nTrain Labels Saved In - {TRAIN_LABELS}")

    val_df.to_csv(VAL_LABELS, index=False)
    print(f"\nValidation Labels Saved In - {VAL_LABELS}")

    test_df.to_csv(TEST_LABELS, index=False)
    print(f"\nTest Labels Saved In - {TEST_LABELS}")

    counts_train = train_df['class'].value_counts().reindex([0, 1], fill_value=0)
    counts_val = val_df['class'].value_counts().reindex([0, 1], fill_value=0)
    counts_test = test_df['class'].value_counts().reindex([0, 1], fill_value=0)

    summary_df = pd.DataFrame(
        {
            'class0': [counts_train[0], counts_val[0], counts_test[0]],
            'class1': [counts_train[1], counts_val[1], counts_test[1]]
        },
        index=['train', 'val', 'test']
    )

    print(f"\n{summary_df}")

    copy_images(train_df, TRAIN_IMAGES_DIR)
    copy_images(val_df, VAL_IMAGES_DIR)
    copy_images(test_df, TEST_IMAGES_DIR)



if __name__ == "__main__":

    # In Percentage
    val_size = 15
    test_size = 15

    split(val_size, test_size)
