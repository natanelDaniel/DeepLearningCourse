import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
import os
import pandas as pd

DATA_PATH = r'Ex2_Ran'

dict_class_name = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                   7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

eps = 1e-12


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


class LogisticRegression:
    def __init__(self, mu=0.01, lr=0.001):
        self.w = np.random.normal(0, 1, (28 * 28, 10))
        self.b = np.random.normal(0, 1, (1, 10))
        self.mu = mu
        self.lr = lr
        self.w_grad = np.zeros((28 * 28, 10))
        self.b_grad = np.zeros((1, 10))

    def forward(self, x):
        """ x is a matrix of shape (n, 28*28) where n is the number of samples"""
        z = x @ self.w + self.b
        return softmax(z)

    def loss(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0] + self.mu * np.sum(self.w ** 2)

    def backward(self, x, y, y_pred):
        self.w_grad = (x.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w
        self.b_grad = np.sum(y_pred - y, axis=0) / y.shape[0]
        self.w -= self.lr * self.w_grad
        self.b -= self.lr * self.b_grad

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

activation_dict = {'relu': relu}

derivative_dict = {'relu': relu_derivative}
class FashionNet:
    def __init__(self, hidden_size=32, mu=0.01, lr=0.001, activation_function='relu', w_1_dropout=0, w_2_dropout=0):
        self.w1 = np.random.normal(0, 1, (28 * 28, hidden_size))
        self.b1 = np.random.normal(0, 1, (1, hidden_size))
        self.w2 = np.random.normal(0, 1, (hidden_size, 10))
        self.b2 = np.random.normal(0, 1, (1, 10))
        self.mu = mu
        self.lr = lr

        self.w1_grad = np.zeros((28 * 28, hidden_size))
        self.b1_grad = np.zeros((1, hidden_size))
        self.w2_grad = np.zeros((hidden_size, 10))
        self.b2_grad = np.zeros((1, 10))
        self.activation_function = activation_dict[activation_function]
        self.activation_derivative = derivative_dict[activation_function]

        self.w_1_dropout = w_1_dropout
        self.w_2_dropout = w_2_dropout

    def dropout(self):
        self.w1 *= np.random.binomial(1, 1 - self.w_1_dropout, self.w1.shape)
        self.w2 *= np.random.binomial(1, 1 - self.w_2_dropout, self.w2.shape)

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        h = self.activation_function(z1)
        z2 = h @ self.w2 + self.b2
        return softmax(z2)

    def loss(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0] + self.mu * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))

    def backward(self, x, y, y_pred):
        z1 = x @ self.w1 + self.b1
        h = self.activation_function(z1)
        self.w2_grad = (h.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w2
        self.b2_grad = np.sum(y_pred - y, axis=0) / y.shape[0]
        self.w1_grad = (x.T @ ((y_pred - y) @ self.w2.T * self.activation_derivative(z1))) / y.shape[0] + 2 * self.mu * self.w1
        self.b1_grad = np.sum((y_pred - y) @ self.w2.T * self.activation_derivative(z1), axis=0) / y.shape[0]

        self.w1 -= self.lr * self.w1_grad
        self.b1 -= self.lr * self.b1_grad
        self.w2 -= self.lr * self.w2_grad
        self.b2 -= self.lr * self.b2_grad


def train(model, x_train, y_train, x_val, y_val, epochs=1000, batch_size=100, verbose=100):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(epochs):
        batch_loss_arr = []
        batch_acc_arr = []
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            # x_batch = data_augmentation(x_batch)
            y_batch = y_train[i:i + batch_size]
            y_pred = model.forward(x_batch)
            y_pred_class = np.argmax(y_pred, axis=1)
            y_batch_class = np.argmax(y_batch, axis=1)
            batch_acc_arr.append(np.mean(y_pred_class == y_batch_class))
            batch_loss_arr.append(model.loss(y_pred, y_batch))

            model.backward(x_batch, y_batch, y_pred)

        train_loss = np.mean(batch_loss_arr)
        train_loss_history.append(train_loss)
        train_acc = np.mean(batch_acc_arr)
        train_acc_history.append(train_acc)

        y_val_pred = model.forward(x_val)
        val_loss = model.loss(y_val_pred, y_val)
        val_loss_history.append(val_loss)
        y_val_pred_class = np.argmax(y_val_pred, axis=1)
        val_acc = np.mean(y_val_pred_class == np.argmax(y_val, axis=1))
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

    lr = 0.001
    epochs = 201
    batch_size = 100
    mu = 0.01
    verbose = 10
    hidden_size = 100
    # model = LogisticRegression(mu=mu, lr=lr)
    model = FashionNet(mu=mu, lr=lr, hidden_size=hidden_size, activation_function='relu')
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(model, x_train, y_train_hot,
                                                                                            x_val, y_val_hot,
                                                                                            epochs=epochs,
                                                                                            batch_size=batch_size,
                                                                                            verbose=verbose)

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
