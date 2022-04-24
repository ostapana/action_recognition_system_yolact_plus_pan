import argparse
import numpy as np
import time
import torch
import random
import matplotlib.pyplot as plt
import csv
import seaborn as sn
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

device = torch.device("cpu")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--path_x', default='', type=str,
                        help='Path to train dataset')
    parser.add_argument('--path_y', default='', type=str,
                        help='Path to true actions for train dataset')
    parser.add_argument('--path_valid_x', default='', type=str,
                        help='Path to the validation dataset')
    parser.add_argument('--path_valid_y', default='', type=str,
                        help='Path to true actions for the validation dataset')
    parser.add_argument('--num_classes', default='', type=str)
    parser.add_argument('--model', default='pretrained/model', type=str,
                        help='Path to model')
    args = parser.parse_args(argv)
    return args


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, path_X, path_y):
        tmp_x = pd.read_csv(path_X)
        n = len(tmp_x)
        tmp_x = tmp_x.values[0:n, 1:6]

        with open(path_y) as f:
            data = f.readlines()

        tmp_y = [int((n.split(',')[1])[0]) for n in data]
        tmp_y = LabelEncoder().fit_transform(tmp_y)

        self.x_data = \
            torch.tensor(tmp_x, dtype=torch.float32).to(device)

        self.y_data = \
            torch.tensor(tmp_y, dtype=torch.int64).to(device)

        self.img_names = [(n.split(',')[0]) for n in data]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        image = self.img_names[idx]
        sample = {
            'predictors': preds,
            'targets': trgts,
            'image': image
        }
        return sample


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.hid1 = torch.nn.Linear(5, 50)
        self.hid2 = torch.nn.Linear(50, 30)
        self.hid3 = torch.nn.Linear(30, 10)
        self.output = torch.nn.Linear(10, num_classes + 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hid1(x)
        x = self.relu(x)
        x = self.hid2(x)
        x = self.relu(x)
        x = self.hid3(x)
        x = self.output(x)
        return x

def get_categories(num_classes=8):
    if num_classes == 8:
        path = 'data/categories/categories8.txt'
    elif num_classes == 4:
        path = 'data/categories/categories4.txt'

    with open(path) as f:
        categories = f.readlines()
    return [f.strip() for f in categories]

def plot_progress(epochs, y_data_train, y_data_valid, y_label, title, label1, label2):
    plt.figure()
    plt.plot(epochs, y_data_train, '-', color='green', label=label1)
    plt.plot(epochs, y_data_valid, '-', color='orange', label=label2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig('visualisation/train/' + title)


def get_index(model, ds, i):
    X = ds[i]['predictors']
    Y = ds[i]['targets']
    with torch.no_grad():
        output = model(X)

    big_idx = torch.argmax(output)
    return Y, big_idx

def accuracy(model, ds):
    # assumes model.eval()
    n_correct = 0
    n_wrong = 0
    for i in range(len(ds)):
        Y, big_idx = get_index(model, ds, i)
        if big_idx == Y:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

def calculate_metrics(model, ds, num_classes=8):
    video_labels = []
    video_pred = []
    with open("data/final_prediction.csv", 'w') as f:
        for i in range(len(ds)):
            Y, big_idx = get_index(model, ds, i)
            video_labels.append(Y.item() + 1)
            video_pred.append(big_idx.item() + 1)

    categories = get_categories(num_classes)
    TTN, TFP, TFN, TTP = 0, 0, 0, 0

    for i in range(1, len(categories) + 1):
        print(categories[i - 1])
        video_labels_c = [0 if n != i else i for n in video_labels]
        video_pred_c = [0 if n != i else i for n in video_pred]
        cf = confusion_matrix(video_labels_c, video_pred_c)
        print(cf.ravel())
        TN, FP, FN, TP = cf.ravel()
        TTN += TN
        TFP += FP
        TFN += FN
        TTP += TP
        print(f"Accuracy: {round((TP + TN) * 100 / (TP + FP + TN + FN), 2)}")
        print(f"Recall: {round(TP * 100 / (TP + FN), 2)}")
        print(f"Precision: {round(TP * 100 / (TP + FP), 2)}")
        print("_"*20 + '\n')

    print(f"Total accuracy {(TTP + TTN) / (TTP + TTN + TFN + TFP)}")
    print(f"Total recall {(TTP) / (TTP + TFN)}")
    print(f"Total precision {(TTP) / (TTP + TFP)}")


def create_confusion_matrix(model, ds, num_classes=8):
    y_pred = []
    y_true = []

    with open("data/final_prediction.csv", 'w') as f:
        writer = csv.writer(f)
        for i in range(len(ds)):
            Y, big_idx = get_index(model, ds, i)
            y_pred.append(big_idx)
            y_true.append(Y)

    categories = get_categories(num_classes)

    cf = confusion_matrix(y_true, y_pred)
    cf2 = [[0 for i in range(0, len(categories))] for i in range(0, len(categories))]
    for i in range(0, len(cf)):
        for j in range(0, len(cf)):
            cf2[i][j] = float(cf[i][j] / sum(cf[i]))
    df_cm = pd.DataFrame(cf2, index=[i for i in categories],
                         columns=[i for i in categories])

    sn.set(font_scale=1.2)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, annot_kws={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig('visualisation/matrix' + str(num_classes) + '.png')

def train(path_X, path_y, path_valid_X, path_valid_y, model_name, num_classes):
    print("\nBegin training \n")

    np.random.seed(1)
    torch.manual_seed(1)

    train_ds = CSVDataset(path_X=path_X, path_y=path_y)
    valid_ds = CSVDataset(path_X=path_valid_X, path_y=path_valid_y)

    bat_size = 8
    train_ldr = torch.utils.data.DataLoader(train_ds,
                                            batch_size=bat_size, shuffle=True)
    valid_ldr = torch.utils.data.DataLoader(valid_ds,
                                            batch_size=bat_size, shuffle=True)

    model = Model(num_classes=num_classes).to(device)

    max_epochs = 300
    ep_log_interval = 50
    lrn_rate = 0.002

    # -----------------------------------------------------------

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate)

    print("\nbatch_size = %3d " % bat_size)
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)

    best_accuracy = 0
    accuracy_valid = 0
    log_interval = 2
    accuracy_progress_train = []
    accuracy_progress_valid = []
    loss_progress_train = []
    loss_progress_valid = []

    epochs = range(0, max_epochs, log_interval)

    # -----------------------------------------------------------

    model.train()
    for epoch in range(0, max_epochs):
        torch.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        epoch_loss_valid = 0  # for one full epoch
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # inputs
            Y = batch['targets']  # shape [10,3] (!)

            optimizer.zero_grad()
            oupt = model(X)  # shape [10] (!)
            loss_val = loss_func(oupt, Y)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages
            loss_val.backward()
            optimizer.step()

        for (batch_idx, batch) in enumerate(valid_ldr):
            X = batch['predictors']  # inputs
            Y = batch['targets']  # shape [10,3] (!)

            optimizer.zero_grad()
            oupt = model(X)
            loss_val = loss_func(oupt, Y)
            epoch_loss_valid += loss_val.item()


        epoch_loss = epoch_loss / len(train_ldr)
        acc_train = accuracy(model, train_ds)


        epoch_loss_valid = epoch_loss_valid / len(valid_ldr)
        acc_valid = accuracy(model, valid_ds)


        if epoch % log_interval == 0:
            loss_progress_train.append(epoch_loss)
            accuracy_progress_train.append(acc_train)
            loss_progress_valid.append(epoch_loss_valid)
            accuracy_progress_valid.append(acc_valid)

        if epoch % ep_log_interval == 0:
            print("\n epoch = %4d   loss = %0.4f \n" % \
                   (epoch, epoch_loss))
        if acc_train > best_accuracy :
            best_accuracy = round(acc_train, 4)
            accuracy_valid = round(acc_valid, 4)
            torch.save(model.state_dict(), model_name + ".pth")
            print(f"SAVED with acc {best_accuracy}, epoch {epoch}")


    # -----------------------------------------------------------

    plot_progress(epochs, loss_progress_train, loss_progress_valid, "Loss",
                  "loss_progress_" + str(max_epochs) + "_" + str(lrn_rate)[2:5] + "_Adam_" + str(num_classes),
                  label1='Training loss', label2='Validation loss')
    plot_progress(epochs, accuracy_progress_train, accuracy_progress_valid, "Recall",
                  "accuracy_progress_" + str(max_epochs) + "_" + str(lrn_rate)[2:5] + "_Adam_" + str(num_classes),
                  label1='Training recall', label2='Validation recall')


    print("Training complete ")
    print("Accuracy on validation data = %0.4f" % accuracy_valid)
    print("Accuracy on training data = %0.4f" % best_accuracy)


def test(path_X, path_y, path_model, num_classes):
    print("\nEvaluation started \n")
    test_ds = CSVDataset(path_X=path_X, path_y=path_y)
    model = Model(num_classes).to(device)
    model.load_state_dict(torch.load(path_model + ".pth"))
    model.eval()
    create_confusion_matrix(model, test_ds, num_classes)
    calculate_metrics(model, test_ds, num_classes)


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args.path_x, args.path_y, args.path_valid_x, args.path_valid_y, args.model, int(args.num_classes))
    elif args.test:
        test(args.path_valid_x, args.path_valid_y, args.model, int(args.num_classes))
