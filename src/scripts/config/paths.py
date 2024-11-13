IDENTIFIER = '2024112-91925'

DOWNLOAD_DIR = 'dataset'
LABELS_DIR = 'labels'
BIN_DIR = 'bin'

DATASET_DIR = f'{DOWNLOAD_DIR}/breast-histopathology-images/IDC_regular_ps50_idx5'

DATASET_LABELS = f'{LABELS_DIR}/dataset_labels.csv'
DATASET_LABELS_WITH_BRIGHTNESS = f'{LABELS_DIR}/dataset_labels_with_brightness.csv'
DATASET_LABELS_FILTERED = f'{LABELS_DIR}/dataset_labels_filtered.csv'
DATASET_LABELS_DISCARDED = f'{LABELS_DIR}/dataset_labels_discarded.csv'

TRAIN_LABELS = f'{LABELS_DIR}/train_labels.csv'
VAL_LABELS = f'{LABELS_DIR}/val_labels.csv'
TEST_LABELS = f'{LABELS_DIR}/test_labels.csv'

TRAIN_IMAGES_DIR = f'{DOWNLOAD_DIR}/train'
VAL_IMAGES_DIR = f'{DOWNLOAD_DIR}/validation'
TEST_IMAGES_DIR = f'{DOWNLOAD_DIR}/test'

MODEL_SAVE_DIR = f'saved_models/{IDENTIFIER}'

PREDICTED_LABELS = f'{MODEL_SAVE_DIR}/predicted_labels.csv'
