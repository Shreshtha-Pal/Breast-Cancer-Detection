import numpy as np
import os
import pandas as pd
import tensorflow as tf
from config.paths import MODEL_SAVE_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS, VAL_IMAGES_DIR
from keras import Input, Model
from keras.src.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.src.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
from utils.get_dataset import dataset_from_directory



def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('leaky_relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('leaky_relu')(x)

    return x

def get_model():
    inputs = Input(shape=(50, 50, 3))

    x = Conv2D(64, kernel_size=(7, 7), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('leaky_relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(4):
        x = residual_block(x, 64)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(4):
        x = residual_block(x, 128)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    plot_model(
        model,
        to_file = f'{MODEL_SAVE_DIR}/model.png',
        show_shapes = True,
        show_dtype= True,
        show_layer_names = True,
        show_layer_activations= True,
        show_trainable = True,
        expand_nested = True
    )

    return model



def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger(f'{MODEL_SAVE_DIR}/model-fit-log.csv')
    scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1)
    return [early_stopping, csv_logger, scheduler]



def get_class_weights():
    train_labels_df = pd.read_csv(TRAIN_LABELS)
    y = train_labels_df['class']
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    print(f'\nClass Weights - {class_weights}\n')
    return class_weights



def train_model(batch_size, epochs):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_ds = dataset_from_directory(TRAIN_IMAGES_DIR, batch_size)
    val_ds = dataset_from_directory(VAL_IMAGES_DIR, batch_size)

    class_weights = get_class_weights()

    callbacks = get_callbacks()

    model = get_model()

    model.fit(x=train_ds, validation_data=val_ds, class_weight=class_weights, callbacks=callbacks, epochs=epochs, verbose=1)

    print("\nSaving Model...")

    model.save(f'{MODEL_SAVE_DIR}/model.keras')

    print(f'\nModel Saved In - {MODEL_SAVE_DIR}')



if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 32
    epochs = 100

    train_model(batch_size, epochs)
