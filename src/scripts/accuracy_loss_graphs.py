import matplotlib.pyplot as plt
import pandas as pd
from config.paths import MODEL_SAVE_DIR



def accuracy_loss_graphs():
    df = pd.read_csv(f'{MODEL_SAVE_DIR}/model-fit-log.csv')

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))

    plt.subplot(1,2,1)
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train accuracy', 'validation accuracy'], loc='best', prop={'size': 12})

    plt.subplot(1,2,2)
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'validation loss'], loc='best', prop={'size': 12})

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    accuracy_loss_graphs()
