def accuracy_loss_graphs():
    with open('src/utils/accuracy_loss_graphs.py') as f:
        code = f.read()
        exec(code, globals())



def train_model():
    with open('src/utils/train_model.py') as f:
        code = f.read()
        exec(code, globals())



def split_dataset():
    with open('src/utils/split_dataset.py') as f:
        code = f.read()
        exec(code, globals())



def filter_dataset():
    with open('src/utils/filter_dataset.py') as f:
        code = f.read()
        exec(code, globals())



def eda():
    with open('src/utils/eda.py') as f:
        code = f.read()
        exec(code, globals())



def label_dataset():
    with open('src/utils/label_dataset.py') as f:
        code = f.read()
        exec(code, globals())



def download_dataset():
    with open('src/utils/download_dataset.py') as f:
        code = f.read()
        exec(code, globals())



if __name__ == "__main__":

    download_dataset()

    label_dataset()

    eda()

    filter_dataset()

    split_dataset()

    train_model()

    accuracy_loss_graphs()
