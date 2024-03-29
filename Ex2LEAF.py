import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns
from LEAF import LEAF

matplotlib.use('TkAgg')

DATA_PATH = r'Ex2_Ran'

dict_class_name = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                   7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

eps = 1e-12


# Todo List:
# 3. add dropout to the model
# 4. check more hyperparameters such as learning rate, mu, hidden size, epochs, batch size and activation function and
#    add your results to the next link: https://docs.google.com/document/d/1GnTlm7aYWlcBHrxH1vFJ-yGBb7Rq-QMaqPoW50D8mFk/edit?usp=sharing
# 6. how meny samples per class we have in the training set
# 7. write the report


def data_augmentation(x, y, max_add=25):
    # Create a mask for values greater than 0.05
    mask = x > 0.05
    random_numbers = np.random.randint(-max_add, max_add+1, size=x.shape[0]) / 255.0

    augmented_data = x + mask * random_numbers[:, np.newaxis]
    augmented_data = np.clip(augmented_data, 0, 1)

    # rand = np.random.rand(augmented_data.shape[0])
    # # add noise to the image if rand > 0.8, just for mask
    # mask = np.logical_and((rand > 0.8)[:, np.newaxis], mask)
    # augmented_data = augmented_data + mask * np.random.normal(0, 0.1, augmented_data.shape)
    # augmented_data = np.clip(augmented_data, 0, 1)
    # flip the image, not for class 5, 7, 9
    rand = np.random.rand(augmented_data.shape[0])
    aug_28_28 = augmented_data.reshape(-1, 28, 28)
    arg_y = np.argmax(y, axis=1)
    mask = np.logical_and(np.logical_and(rand > 0.5, arg_y != 5), np.logical_and(arg_y != 7, arg_y != 9))
    aug_28_28[mask] = np.flip(aug_28_28[mask], axis=2)

    # shift randomly the image
    shift_vertical = np.random.randint(-1, 2, aug_28_28.shape[0])
    shift_horizontal = np.random.randint(-1, 2, aug_28_28.shape[0])
    for i in range(aug_28_28.shape[0]):
        aug_28_28[i] = np.roll(aug_28_28[i], shift_vertical[i], axis=0)
        aug_28_28[i] = np.roll(aug_28_28[i], shift_horizontal[i], axis=1)

    aug_28_28[shift_vertical == 1][:, 0] = 0
    aug_28_28[shift_vertical == -1][:, -1] = 0
    aug_28_28[shift_horizontal == 1][:, :, 0] = 0
    aug_28_28[shift_horizontal == -1][:, :, -1] = 0
    # aug_28_28[shift_horizontal == 2][:, :, 0:2] = 0
    # aug_28_28[shift_horizontal == -2][:, :, -2:] = 0

    augmented_data = aug_28_28.reshape(-1, 28 * 28)
    return augmented_data


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


def visualize(x_train, y_train, h=4):
    #     visualize the data, 4*10 subplot, 4 row, 10 column, 4 per class
    fig, ax = plt.subplots(h, 10, figsize=(10, 4))
    for j in range(10):
        for i in range(h):
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
    def __init__(self, mu=0.01, lr=0.001, initialization='kaiming', momentum=None, beta1=0.9, beta2=0.999, lr_decay=1):
        if initialization == 'kaiming':
            self.w = np.random.normal(0, np.sqrt(2 / (28 * 28)), (28 * 28, 10))
            self.b = np.zeros((1, 10))
        else:
            self.w = np.random.normal(0, 1, (28 * 28, 10))
            self.b = np.random.normal(0, 1, (1, 10))

        self.mu = mu
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_grad = np.zeros((28 * 28, 10))
        self.b_grad = np.zeros((1, 10))
        self.m_t_w = np.zeros_like(self.w)
        self.v_t_w = np.zeros_like(self.w)
        self.m_t_b = np.zeros_like(self.b)
        self.v_t_b = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        z = x @ self.w + self.b
        return self.softmax(z)

    def loss(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0] + self.mu * np.sum(self.w ** 2)

    def backward(self, x, y, y_pred):
        self.t += 1
        self.w_grad = (x.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w
        self.b_grad = np.sum(y_pred - y, axis=0) / y.shape[0]

        if self.momentum is not None:
            self.w_grad = self.momentum * self.m_t_w + (1 - self.momentum) * self.w_grad
            self.b_grad = self.momentum * self.m_t_b + (1 - self.momentum) * self.b_grad

        self.m_t_w = self.beta1 * self.m_t_w + (1 - self.beta1) * self.w_grad
        self.v_t_w = self.beta2 * self.v_t_w + (1 - self.beta2) * (self.w_grad ** 2)
        self.m_t_b = self.beta1 * self.m_t_b + (1 - self.beta1) * self.b_grad
        self.v_t_b = self.beta2 * self.v_t_b + (1 - self.beta2) * (self.b_grad ** 2)

        m_t_w_hat = self.m_t_w / (1 - self.beta1 ** self.t)
        v_t_w_hat = self.v_t_w / (1 - self.beta2 ** self.t)
        m_t_b_hat = self.m_t_b / (1 - self.beta1 ** self.t)
        v_t_b_hat = self.v_t_b / (1 - self.beta2 ** self.t)

        self.w -= self.lr * m_t_w_hat / (np.sqrt(v_t_w_hat) + eps)
        self.b -= self.lr * m_t_b_hat / (np.sqrt(v_t_b_hat) + eps)

    def update_lr(self):
        self.lr *= self.lr_decay

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x >= 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def silu_derivative(x):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))


activation_dict = {'relu': relu, 'silu': silu}

derivative_dict = {'relu': relu_derivative, 'silu': silu_derivative}


class FashionNet:
    def __init__(self, hidden_size=32, mu=0.01, lr=0.001, activation_function='relu', w_1_dropout=0, w_2_dropout=0,
                 initialzation='kaiming', momentum=None, beta1=0.9, beta2=0.999, lr_decay=1, factor_init=1):
        if initialzation == 'kaiming':
            self.w1 = np.random.normal(0, np.sqrt(2 / (28 * 28 * factor_init)), (28 * 28, hidden_size))
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.normal(0, np.sqrt(2 / hidden_size * factor_init), (hidden_size, 10))
            self.b2 = np.zeros((1, 10))
        else:
            self.w1 = np.random.normal(0, 1, (28 * 28, hidden_size))
            self.b1 = np.random.normal(0, 1, (1, hidden_size))
            self.w2 = np.random.normal(0, 1, (hidden_size, 10))
            self.b2 = np.random.normal(0, 1, (1, 10))

        self.mu = mu
        self.lr = lr
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.w1_grad = np.zeros((28 * 28, hidden_size))
        self.b1_grad = np.zeros((1, hidden_size))
        self.w2_grad = np.zeros((hidden_size, 10))
        self.b2_grad = np.zeros((1, 10))
        self.activation_function = LEAF(activation_function, size=(1, hidden_size), lr_p1=0.01, lr_p2=0.01, lr_p3=0.01,
                                        lr_p4=0.01)

        self.w_1_dropout = w_1_dropout
        self.w_2_dropout = w_2_dropout

        self.prev_w1_grad = np.zeros_like(self.w1)
        self.prev_b1_grad = np.zeros_like(self.b1)
        self.prev_w2_grad = np.zeros_like(self.w2)
        self.prev_b2_grad = np.zeros_like(self.b2)

        self.beta1 = beta1
        self.beta2 = beta2
        self.m_t_w1 = np.zeros_like(self.w1)
        self.v_t_w1 = np.zeros_like(self.w1)
        self.m_t_b1 = np.zeros_like(self.b1)
        self.v_t_b1 = np.zeros_like(self.b1)
        self.m_t_w2 = np.zeros_like(self.w2)
        self.v_t_w2 = np.zeros_like(self.w2)
        self.m_t_b2 = np.zeros_like(self.b2)
        self.v_t_b2 = np.zeros_like(self.b2)
        self.t = 0

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
        self.t += 1
        z1 = x @ self.w1 + self.b1

        self.activation_function.update(z1, (y_pred - y) @ self.w2.T * self.activation_function.derivative(z1))

        h = self.activation_function(z1)

        self.w2_grad = (h.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w2
        self.b2_grad = np.sum(y_pred - y, axis=0) / y.shape[0]

        w2_delta = (y_pred - y) @ self.w2.T * self.activation_function.derivative(z1)
        self.w1_grad = (x.T @ w2_delta) / y.shape[0] + 2 * self.mu * self.w1
        self.b1_grad = np.sum(w2_delta, axis=0) / y.shape[0]

        if self.momentum is not None:
            self.w1_grad = self.momentum * self.prev_w1_grad + (1 - self.momentum) * self.w1_grad
            self.b1_grad = self.momentum * self.prev_b1_grad + (1 - self.momentum) * self.b1_grad
            self.w2_grad = self.momentum * self.prev_w2_grad + (1 - self.momentum) * self.w2_grad
            self.b2_grad = self.momentum * self.prev_b2_grad + (1 - self.momentum) * self.b2_grad

        # Adam optimization
        self.m_t_w1 = self.beta1 * self.m_t_w1 + (1 - self.beta1) * self.w1_grad
        self.v_t_w1 = self.beta2 * self.v_t_w1 + (1 - self.beta2) * (self.w1_grad ** 2)
        self.m_t_b1 = self.beta1 * self.m_t_b1 + (1 - self.beta1) * self.b1_grad
        self.v_t_b1 = self.beta2 * self.v_t_b1 + (1 - self.beta2) * (self.b1_grad ** 2)

        self.m_t_w2 = self.beta1 * self.m_t_w2 + (1 - self.beta1) * self.w2_grad
        self.v_t_w2 = self.beta2 * self.v_t_w2 + (1 - self.beta2) * (self.w2_grad ** 2)
        self.m_t_b2 = self.beta1 * self.m_t_b2 + (1 - self.beta1) * self.b2_grad
        self.v_t_b2 = self.beta2 * self.v_t_b2 + (1 - self.beta2) * (self.b2_grad ** 2)

        m_t_w1_hat = self.m_t_w1 / (1 - self.beta1 ** self.t)
        v_t_w1_hat = self.v_t_w1 / (1 - self.beta2 ** self.t)
        m_t_b1_hat = self.m_t_b1 / (1 - self.beta1 ** self.t)
        v_t_b1_hat = self.v_t_b1 / (1 - self.beta2 ** self.t)

        m_t_w2_hat = self.m_t_w2 / (1 - self.beta1 ** self.t)
        v_t_w2_hat = self.v_t_w2 / (1 - self.beta2 ** self.t)
        m_t_b2_hat = self.m_t_b2 / (1 - self.beta1 ** self.t)
        v_t_b2_hat = self.v_t_b2 / (1 - self.beta2 ** self.t)

        # Update parameters
        self.w1 -= self.lr * m_t_w1_hat / (np.sqrt(v_t_w1_hat) + eps)
        self.b1 -= self.lr * m_t_b1_hat / (np.sqrt(v_t_b1_hat) + eps)
        self.w2 -= self.lr * m_t_w2_hat / (np.sqrt(v_t_w2_hat) + eps)
        self.b2 -= self.lr * m_t_b2_hat / (np.sqrt(v_t_b2_hat) + eps)

        # Update previous gradients for momentum
        if self.momentum is not None:
            self.prev_w1_grad = self.w1_grad.copy()
            self.prev_b1_grad = self.b1_grad.copy()
            self.prev_w2_grad = self.w2_grad.copy()
            self.prev_b2_grad = self.b2_grad.copy()

    def update_lr(self):
        self.lr *= self.lr_decay


def train(model, x_train, y_train, x_val, y_val, epochs=1000, batch_size=100, verbose=100, batch_size_add=0, max_add=25):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(epochs):
        batch_loss_arr = []
        batch_acc_arr = []
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            x_batch = data_augmentation(x_batch, y_batch, max_add)

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
            print(np.mean(model.activation_function.p1()), np.mean(model.activation_function.p2()), np.mean(model.activation_function.p3()), np.mean(model.activation_function.p4()))
            print('epoch {}, train loss {}, val loss {}, train acc {}, val acc {}'.format(epoch, train_loss,
                                                                                          val_loss, train_acc, val_acc))

        model.update_lr()
        batch_size += batch_size_add
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val = load_data()
    # visualize(x_train, y_train, h=10)

    # data normalization
    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255


    # for classes 0,2,4,6 -> duplicate the samples
    class_0 = x_train[y_train == 0]
    class_2 = x_train[y_train == 2]
    class_4 = x_train[y_train == 4]
    class_6 = x_train[y_train == 6]

    x_train = np.concatenate([x_train, class_0, class_2, class_4, class_6, class_6], axis=0)
    y_train = np.concatenate([y_train, np.zeros(class_0.shape[0]), 2 * np.ones(class_2.shape[0]), 4 * np.ones(class_4.shape[0]), 6 * np.ones(class_6.shape[0]), 6 * np.ones(class_6.shape[0])]).astype(int)

    # shuffle the data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    y_train_hot = one_hot_encoding(y_train)
    y_val_hot = one_hot_encoding(y_val)

    lr = 0.001
    lr_decay = 0.99
    epochs = 251
    batch_size = 64
    mu = 0.00005
    verbose = 1
    hidden_size = 128
    momentum = None
    batch_size_add = 1
    factor_init = 4
    max_add = 5
    # model = LogisticRegression(mu=mu, lr=lr, lr_decay=lr_decay)
    model = FashionNet(mu=mu, lr=lr, hidden_size=hidden_size, activation_function='relu', momentum=momentum,
                       lr_decay=lr_decay, factor_init=factor_init, beta1=0.9, beta2=0.999)
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(model, x_train, y_train_hot,
                                                                                            x_val, y_val_hot,
                                                                                            epochs=epochs,
                                                                                            batch_size=batch_size,
                                                                                            verbose=verbose,
                                                                                            batch_size_add=batch_size_add,
                                                                                            max_add=max_add)


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

    y_pred = model.forward(x_val)
    arg_y = np.argmax(y_pred, axis=1)
    # Step 2: Compute Accuracy for Each Class
    accuracy_per_class = []
    for i in range(10):  # Assuming you have 10 classes
        indices = y_val == i
        accuracy_i = accuracy_score(y_val[indices], arg_y[indices])
        accuracy_per_class.append(accuracy_i)

    # Print accuracy for each class
    for i, acc in enumerate(accuracy_per_class):
        print(f'Class {dict_class_name[i]}: {acc:.2f}')

    # Alternatively, use classification report for more details
    print("Classification Report:")
    print(classification_report(y_val, arg_y))

    # Step 3: Generate Confusion Matrix
    conf_matrix = confusion_matrix(y_val, arg_y)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=dict_class_name.values(),
                yticklabels=dict_class_name.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

#     lets show the wrong samples of class 6

    wrong_samples = x_val[(y_val == 6) & (arg_y != 6)]
    fig, ax = plt.subplots(1, 10, figsize=(10, 4))
    for i in range(10):
        ax[i].imshow(wrong_samples[i].reshape(28, 28), cmap='gray')
        ax[i].set_title(dict_class_name[arg_y[(y_val == 6) & (arg_y != 6)][i]])
        ax[i].axis('off')

    plt.show()

    good_samples = x_val[(y_val == 6) & (arg_y == 6)]
    fig, ax = plt.subplots(1, 10, figsize=(10, 4))
    for i in range(10):
        ax[i].imshow(good_samples[i].reshape(28, 28), cmap='gray')
        ax[i].set_title(dict_class_name[arg_y[(y_val == 6) & (arg_y == 6)][i]])
        ax[i].axis('off')

    plt.show()
