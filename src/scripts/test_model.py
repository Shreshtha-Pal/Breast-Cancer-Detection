import numpy as np
import pandas as pd
from config.paths import MODEL_SAVE_DIR, PREDICTED_LABELS, TEST_IMAGES_DIR, TEST_LABELS
from keras.src.saving import load_model
from utils.get_dataset import dataset_from_directory



def test_model(batch_size):
    model = load_model(f'{MODEL_SAVE_DIR}/model.keras')

    test_ds = dataset_from_directory(TEST_IMAGES_DIR, batch_size)

    predictions = model.predict(test_ds)

    pred_rows = []
    conf_rows = []

    for prediction in predictions:
        pred_rows.append(np.argmax(prediction))
        conf_rows.append(np.max(prediction))

    test_df = pd.read_csv(TEST_LABELS)
    pred_df = test_df.copy()

    pred_df['pred'] = pred_rows
    pred_df['conf'] = conf_rows

    pred_df.to_csv(PREDICTED_LABELS, index=False)



if __name__ == '__main__':

    batch_size = 32

    test_model(batch_size)
