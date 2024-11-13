import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from config.paths import DATASET_LABELS, DATASET_LABELS_WITH_BRIGHTNESS
from tqdm import tqdm



def image_mean_brightness(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel = lab_image[:, :, 0]
    mean_brightness = np.mean(l_channel)
    return mean_brightness

def calc_brightness():
    dataset_labels_df = pd.read_csv(DATASET_LABELS)

    dataset_labels_df['brightness'] = None

    dataset_labels_with_brightness_df = dataset_labels_df.copy()

    for index, row in tqdm(dataset_labels_with_brightness_df.iterrows(), total=len(dataset_labels_with_brightness_df), desc=f"Calculating Brightness Of Each Image"):

        image_path = f"{row['dir']}/{row['image']}"
        image = cv2.imread(image_path)
        mean_brightness = image_mean_brightness(image)

        dataset_labels_with_brightness_df.at[index, 'brightness'] = mean_brightness

    dataset_labels_with_brightness_df.to_csv(DATASET_LABELS_WITH_BRIGHTNESS, index=False)

    print(f"\nLabels With Brightness Saved In - {DATASET_LABELS_WITH_BRIGHTNESS}")

def plot_brightness_hist():
    if (not os.path.isfile(DATASET_LABELS_WITH_BRIGHTNESS)):
        calc_brightness()

    dataset_labels_with_brightness_df = pd.read_csv(DATASET_LABELS_WITH_BRIGHTNESS)

    brightness_values_class_0 = dataset_labels_with_brightness_df[dataset_labels_with_brightness_df['class'] == 0]['brightness']
    brightness_values_class_1 = dataset_labels_with_brightness_df[dataset_labels_with_brightness_df['class'] == 1]['brightness']

    plt.figure(figsize=(20, 7))

    plt.subplot(1, 2, 1)
    plt.hist(brightness_values_class_0, bins=64, color='lightblue', edgecolor='black')
    plt.title('Brightness Distribution for Class 0')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xticks(np.arange(0, 255, 10), rotation=90)
    plt.yticks(np.arange(0, 15000, 500))

    plt.subplot(1, 2, 2)
    plt.hist(brightness_values_class_1, bins=64, color='lightgreen', edgecolor='black')
    plt.title('Brightness Distribution for Class 1')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xticks(np.arange(0, 255, 10), rotation=90)
    plt.yticks(np.arange(0, 4750, 500))

    plt.tight_layout()
    plt.show()



def plot_dataset_distribution():
    dataset_labels_df = pd.read_csv(DATASET_LABELS)

    class_counts = dataset_labels_df['class'].value_counts()

    plt.figure(figsize=(7, 6))

    colormap = plt.get_cmap('tab10')
    colors = colormap([i for i in range(len(class_counts.index))])

    bars = plt.bar(class_counts.index, class_counts.values, color=colors, tick_label=class_counts.index)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    plt.title("Dataset Distribution")
    plt.xlabel("Class")
    plt.ylabel("Image Count")

    legend_labels = ["IDC -ve", "IDC +ve"]
    legend_colors = [colors[i] for i in range(len(colors))]
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    legend = plt.legend(legend_patches, legend_labels, title="Class Labels", loc='upper right', frameon=True)
    plt.gca().add_artist(legend)

    plt.tight_layout()
    plt.show()



def visualize_dataset():
    dataset_labels_df = pd.read_csv(DATASET_LABELS)

    x = 3
    y = 3

    images_per_class = int(x * y)

    label_0_images = dataset_labels_df[dataset_labels_df['class'] == 0].sample(n=images_per_class)
    label_1_images = dataset_labels_df[dataset_labels_df['class'] == 1].sample(n=images_per_class)

    _, axes = plt.subplots(y, x*2, figsize=(x*3, y*1.5))

    for ax in axes.flatten():
        ax.axis('off')

    for i, (_, row) in enumerate(label_0_images.iterrows()):
        img_path = f"{row['dir']}/{row['image']}"
        img = mpimg.imread(img_path)
        axes[i//x, i%x].imshow(img)

    for i, (_, row) in enumerate(label_1_images.iterrows()):
        img_path = f"{row['dir']}/{row['image']}"
        img = mpimg.imread(img_path)
        axes[i//x, x+(i%x)].imshow(img)

    axes[0, 1].set_title('IDC -ve', fontsize=14)
    axes[0, x+1].set_title('IDC +ve', fontsize=14)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    visualize_dataset()

    plot_dataset_distribution()

    plot_brightness_hist()
