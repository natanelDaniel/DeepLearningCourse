import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
import torch.nn as nn
matplotlib.use('TkAgg')

DATA_PATH = r'Ex2_Ran'

dict_class_name = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                   7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

eps = 1e-12

# Todo List:
# 1. read again the derivative, check that the derivative is correct
# 2. build model with pytorch by using the same architecture and check if the results are the same
# 3. add dropout to the model
# 4. check more hyperparameters such as learning rate, mu, hidden size, epochs, batch size and activation function and
#    add your results to the next link: https://docs.google.com/document/d/1GnTlm7aYWlcBHrxH1vFJ-yGBb7Rq-QMaqPoW50D8mFk/edit?usp=sharing
# 5. which samples we are wrong, which samples we are right, which classes we are wrong, which classes we are right
# 6. how meny samples per class we have in the training set
# 7. write the report


def data_augmentation(x):
    x_28_28 = x.reshape(-1, 28, 28)
    shift_vertical = np.random.randint(-1, 1)
    shift_horizontal = np.random.randint(-1, 1)
    x_28_28 = np.roll(x_28_28, shift_vertical, axis=0)
    x_28_28 = np.roll(x_28_28, shift_horizontal, axis=1)
    return x_28_28.reshape(-1, 28 * 28)


def load_data():
    #     load csv file
    train_data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    # convert to numpy array
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    train_labels = train_data[:, 0]
    train_data = train_data[:, 1:]

    # train test split
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1,
                                                                      random_state=42, stratify=train_labels,
                                                                      shuffle=True)

    return train_data, val_data, test_data, train_labels, val_labels


def visualize(x_train, y_train):
    #     visualize the data, 4*10 subplot, 4 row, 10 column, 4 per class
    fig, ax = plt.subplots(4, 10, figsize=(10, 4))
    for j in range(10):
        for i in range(4):
            ax[i, j].imshow(x_train[y_train == j][i].reshape(28, 28), cmap='gray')
            ax[i, j].set_title(dict_class_name[j])
            ax[i, j].axis('off')

    plt.show()


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtracting max for numerical stability
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + eps)


def one_hot_encoding(y):
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


class LogisticRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


class FashionNet(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.activation_function = nn.ReLU()

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def train(model, x_train, y_train, x_val, y_val, epochs=1000, batch_size=100, verbose=100, lr=0.001):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        batch_loss_arr = []
        batch_acc_arr = []
        for i in range(0, x_train.shape[0], batch_size):
            optimizer.zero_grad()
            x_batch = x_train[i:i + batch_size]
            torch_x_batch = torch.from_numpy(x_batch).float()
            # x_batch = data_augmentation(x_batch)
            y_batch = y_train[i:i + batch_size]
            torch_y_batch = torch.from_numpy(y_batch).long()
            y_pred = model.forward(torch_x_batch)
            y_pred_class = torch.argmax(y_pred, dim=1)
            y_batch_class = torch.argmax(torch_y_batch, dim=1)
            batch_acc_arr.append(torch.mean((y_pred_class == y_batch_class).float()).item())
            loss = criterion(y_pred, torch_y_batch.float())
            batch_loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(batch_loss_arr)
        train_loss_history.append(train_loss)
        train_acc = np.mean(batch_acc_arr)
        train_acc_history.append(train_acc)

        torch_x_val = torch.from_numpy(x_val).float()

        y_val_pred = model.forward(torch_x_val)
        val_loss = criterion(y_val_pred, torch.from_numpy(y_val).float()).item()
        val_loss_history.append(val_loss)
        y_val_pred_class = torch.argmax(y_val_pred, dim=1)
        val_acc = torch.mean((y_val_pred_class == torch.argmax(torch.from_numpy(y_val), dim=1)).float()).item()
        val_acc_history.append(val_acc)
        if epoch % verbose == 0:
            print('epoch {}, train loss {}, val loss {}, train acc {}, val acc {}'.format(epoch, train_loss,
                                                                                          val_loss, train_acc, val_acc))
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val = load_data()
    # visualize(x_train, y_train)

    # data normalization
    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255

    y_train_hot = one_hot_encoding(y_train)
    y_val_hot = one_hot_encoding(y_val)

    lr = 0.002
    epochs = 201
    batch_size = 100
    mu = 0.01
    verbose = 10
    hidden_size = 100
    # model = LogisticRegression(mu=mu, lr=lr)
    model = LogisticRegression()
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(model, x_train, y_train_hot,
                                                                                            x_val, y_val_hot,
                                                                                            epochs=epochs,
                                                                                            batch_size=batch_size,
                                                                                            verbose=verbose, lr=lr)

    plt.figure()
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc_history, label='train acc')
    plt.plot(val_acc_history, label='val acc')
    plt.legend()
    plt.show()

    y_test_pred = model.forward(x_test)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)
