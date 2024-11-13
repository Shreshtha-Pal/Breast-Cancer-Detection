from keras.src.utils import image_dataset_from_directory



def normalize_dataset(image, label):
    image = image/255
    return image, label

def dataset_from_directory(images_dir, batch_size):
    ds = image_dataset_from_directory(
        images_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(50, 50),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    ds = ds.map(normalize_dataset)
    return ds
