from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def plot_confusion_matrix(conf_matrix):

    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f'{n} - {p}\n\n{c}' for n, c, p in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues',
                        xticklabels=['IDC (-)', 'IDC (+)'],
                        yticklabels=['IDC (-)', 'IDC (+)'])

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix', y=-0.1)

    plt.show()



def get_scores(pred_csv, show_confusion_matrix):

    df = pd.read_csv(pred_csv)

    y_true = df['class']
    y_pred = df['pred']

    print(classification_report(y_true, y_pred))

    if show_confusion_matrix:

        conf_matrix = confusion_matrix(y_true, y_pred)

        plot_confusion_matrix(conf_matrix)



if __name__ == '__main__':

    pred_csv = r'saved_models/2024112-91925/predicted_labels.csv'

    show_confusion_matrix = True

    get_scores(pred_csv, show_confusion_matrix)
