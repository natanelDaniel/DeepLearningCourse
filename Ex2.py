from copy import deepcopy
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns

matplotlib.use('TkAgg')

# define the data path:
DATA_PATH = r'Ex2_Ran'

# save the class names in a dictionary: (keys are the class number, values are the class name)
dict_class_name = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                   3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                   7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# define a small number to avoid division by zero and get numerical stability
eps = 1e-12


def data_augmentation(x, y, max_add=25, random_shift=1):
    """
    Augment the data by adding random numbers to the pixels of the image
    :param x:  the data
    :param y: the labels
    :param max_add: the maximum number to add to the pixels (uniformly)
    :param random_shift: the maximum number to shift the image (vertically and horizontally)
    :return: the augmented data
    """
    # Create a mask for values greater than 0.05
    mask = x > 0.05
    random_numbers = np.random.randint(-max_add, max_add+1, size=x.shape[0]) / 255.0

    augmented_data = x + mask * random_numbers[:, np.newaxis]
    augmented_data = np.clip(augmented_data, 0, 1)

    # flip the image, not for class 5, 7, 9
    rand = np.random.rand(augmented_data.shape[0])
    aug_28_28 = augmented_data.reshape(-1, 28, 28)
    arg_y = np.argmax(y, axis=1)
    mask = np.logical_and(np.logical_and(rand > 0.5, arg_y != 5), np.logical_and(arg_y != 7, arg_y != 9))
    aug_28_28[mask] = np.flip(aug_28_28[mask], axis=2)

    # shift randomly the image
    if random_shift != 0:
        shift_vertical = np.random.randint(-random_shift, random_shift+1, aug_28_28.shape[0])
        shift_horizontal = np.random.randint(-random_shift, random_shift+1, aug_28_28.shape[0])
        for i in range(aug_28_28.shape[0]):
            aug_28_28[i] = np.roll(aug_28_28[i], shift_vertical[i], axis=0)
            aug_28_28[i] = np.roll(aug_28_28[i], shift_horizontal[i], axis=1)

        aug_28_28[shift_vertical == 1][:, 0] = 0
        aug_28_28[shift_vertical == -1][:, -1] = 0
        aug_28_28[shift_horizontal == 1][:, :, 0] = 0
        aug_28_28[shift_horizontal == -1][:, :, -1] = 0

    augmented_data = aug_28_28.reshape(-1, 28 * 28)
    return augmented_data


def load_data():
    """
    Load the data from the csv files
    :return:  the train data, validation data, test data, train labels, validation labels
    """
    # load csv file
    train_data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    # convert to numpy array
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    train_labels = train_data[:, 0]
    train_data = train_data[:, 1:]

    # train test split
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                      test_size=0.1, random_state=42,
                                                                      stratify=train_labels, shuffle=True)
    return train_data, val_data, test_data, train_labels, val_labels


def visualize(x_train, y_train, h=4):
    """
    Visualize the data using matplotlib.pyplot and plot it in a subplot matrix
    :param x_train: the data
    :param y_train: the labels
    :param h: the number of samples to visualize per class
    :return: None
    """
    # visualize the data, 4*10 subplot, 4 row, 10 column, 4 per class
    fig, ax = plt.subplots(h, 10, figsize=(10, 4))
    for j in range(10):
        ax[0, j].set_title(dict_class_name[j])
        for i in range(h):
            ax[i, j].imshow(x_train[y_train == j][i].reshape(28, 28), cmap='gray')
            ax[i, j].axis('off')
    plt.show()


def softmax(x):
    """
    Compute the softmax of the input matrix
    :param x: the input matrix
    :return: the softmax of the input matrix
    """
    # Subtracting max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + eps)


def one_hot_encoding(y):
    """
    One hot encode the labels
    :param y: original labels with values from 0 to 9 as a vector
    :return: the one hot encoded labels
    """
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


class LogisticRegression:
    """
    This class represents a logistic regression model
    """
    def __init__(self, mu=0.01, lr=0.001, initialization='kaiming', momentum=None, beta1=0.9, beta2=0.999, lr_decay=1, factor_init=1):
        if initialization == 'kaiming':
            self.w = np.random.normal(0, np.sqrt(2 / (28 * 28 * factor_init)), (28 * 28, 10))
            self.b = np.zeros((1, 10))
        else:
            self.w = np.random.normal(0, 1, (28 * 28, 10))
            self.b = np.random.normal(0, 1, (1, 10))

        # define the hyperparameters and the model parameters
        self.mu = mu
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        # Adam optimization parameters
        self.beta1 = beta1
        self.beta2 = beta2
        # define the model gradients
        self.w_grad = np.zeros((28 * 28, 10))
        self.b_grad = np.zeros((1, 10))
        # define the previous gradients for momentum
        self.m_t_w = np.zeros_like(self.w)
        self.v_t_w = np.zeros_like(self.w)
        self.m_t_b = np.zeros_like(self.b)
        self.v_t_b = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x, training=False):
        """
        Forward pass of the model
        :param x: the input data
        :param training: a flag to indicate if the model is in training mode
        :return: the output of the model
        """
        z = x @ self.w + self.b
        return self.softmax(z)

    def loss(self, y_pred, y):
        """
        Compute the loss of the model
        :param y_pred: the output of the model
        :param y: the labels
        :return: the loss of the model
        """
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0] + self.mu * np.sum(self.w ** 2)

    def backward(self, x, y, y_pred):
        """
        Backward pass of the model
        :param x: input data
        :param y: labels
        :param y_pred: the output of the model
        :return: None
        """
        self.t += 1
        self.w_grad = (x.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w
        self.b_grad = np.sum(y_pred - y, axis=0) / y.shape[0]

        # Adam optimization
        if self.momentum is not None:
            self.w_grad = self.momentum * self.m_t_w + (1 - self.momentum) * self.w_grad
            self.b_grad = self.momentum * self.m_t_b + (1 - self.momentum) * self.b_grad

        self.m_t_w = self.beta1 * self.m_t_w + (1 - self.beta1) * self.w_grad
        self.v_t_w = self.beta2 * self.v_t_w + (1 - self.beta2) * (self.w_grad ** 2)
        self.m_t_b = self.beta1 * self.m_t_b + (1 - self.beta1) * self.b_grad
        self.v_t_b = self.beta2 * self.v_t_b + (1 - self.beta2) * (self.b_grad ** 2)

        # Update parameters
        m_t_w_hat = self.m_t_w / (1 - self.beta1 ** self.t)
        v_t_w_hat = self.v_t_w / (1 - self.beta2 ** self.t)
        m_t_b_hat = self.m_t_b / (1 - self.beta1 ** self.t)
        v_t_b_hat = self.v_t_b / (1 - self.beta2 ** self.t)

        # Update parameters
        self.w -= self.lr * m_t_w_hat / (np.sqrt(v_t_w_hat) + eps)
        self.b -= self.lr * m_t_b_hat / (np.sqrt(v_t_b_hat) + eps)

    def update_lr(self):
        """
        Update the learning rate of the model using the learning rate decay
        :return: None
        """
        self.lr *= self.lr_decay

    def softmax(self, x):
        """
        Compute the softmax of the input matrix
        :param x: the input matrix
        :return: the softmax of the input matrix
        """
        # Subtracting max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FashionNet:
    # This class represents a neural network model
    def __init__(self, hidden_size=32, mu=0.01, lr=0.001, activation_function='relu', dropout_rate=0,
                 initialzation='kaiming', momentum=None, beta1=0.9, beta2=0.999, lr_decay=1, factor_init=1):
        """
        This function initializes the model parameters and the model itself
        :param hidden_size: the size of the hidden layer
        :param mu: the regularization hyperparameter (if used)
        :param lr: the learning rate
        :param activation_function: the activation function to use
        :param dropout_rate: the dropout rate
        :param initialzation: the initialization method to use
        :param momentum: the momentum hyperparameter (if used)
        :param beta1: an Adam optimization hyperparameter
        :param beta2: an Adam optimization hyperparameter
        :param lr_decay: the learning rate decay hyperparameter
        :param factor_init: the factor to use in the initialization
        """
        # initialize the model parameters using the kaiming initialization method
        if initialzation == 'kaiming':
            self.w1 = np.random.normal(0, np.sqrt(2 / (28 * 28 * factor_init)), (28 * 28, hidden_size))
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.normal(0, np.sqrt(2 / hidden_size * factor_init), (hidden_size, 10))
            self.b2 = np.zeros((1, 10))
        # initialize the model parameters using the normal initialization method
        else:
            self.w1 = np.random.normal(0, 1, (28 * 28, hidden_size))
            self.b1 = np.random.normal(0, 1, (1, hidden_size))
            self.w2 = np.random.normal(0, 1, (hidden_size, 10))
            self.b2 = np.random.normal(0, 1, (1, 10))

        # define the hyperparameters and the model parameters
        self.mu = mu
        self.lr = lr
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.w1_grad = np.zeros((28 * 28, hidden_size))
        self.b1_grad = np.zeros((1, hidden_size))
        self.w2_grad = np.zeros((hidden_size, 10))
        self.b2_grad = np.zeros((1, 10))

        # define the activation functions and their derivative dictionaries
        self.activation_dict = {'relu': self.relu, 'silu': self.silu, 'leaky_relu': self.leaky_relu}
        self.derivative_dict = {'relu': self.relu_derivative, 'silu': self.silu_derivative, 'leaky_relu': self.leaky_relu_derivative}

        # choose the activation function and its derivative according to the input
        self.activation_function = self.activation_dict[activation_function]
        self.activation_derivative = self.derivative_dict[activation_function]

        # define the dropout rate
        self.dropout_rate = dropout_rate

        # define the previous gradients for momentum
        self.prev_w1_grad = np.zeros_like(self.w1)
        self.prev_b1_grad = np.zeros_like(self.b1)
        self.prev_w2_grad = np.zeros_like(self.w2)
        self.prev_b2_grad = np.zeros_like(self.b2)

        # Adam optimization parameters
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

    def relu(self, x):
        """
        This function calculates the ReLU function for X
        :param x: input data
        :return: the ReLU function for X
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        This function calculates the derivative of the ReLU function for X
        :param x: input data
        :return: the derivative of the ReLU function for X
        """
        return np.where(x >= 0, 1, 0)

    def sigmoid(self, x):
        """
        This function calculates the sigmoid function for X
        :param x:  input data
        :return: the sigmoid function for X
        """
        return 1 / (1 + np.exp(-x))

    def silu(self, x):
        """
        This function calculates the SiLU function for X which is defined as X * sigmoid(X)
        :param x: input data
        :return: the SiLU function for X
        """
        return x * self.sigmoid(x)

    def leaky_relu(self, x):
        """
        This function calculates the Leaky ReLU function for X
        :param x: input data
        :return: the Leaky ReLU function for X
        """
        return np.maximum(0.01 * x, x)

    def silu_derivative(self, x):
        """
        This function calculates the derivative of the SiLU function for X
        :param x: input data
        :return: the derivative of the SiLU function for X
        """
        return self.sigmoid(x) + x * self.sigmoid(x) * (1 - self.sigmoid(x))

    def leaky_relu_derivative(self, x):
        """
        This function calculates the derivative of the Leaky ReLU function for X
        :param x: input data
        :return: the derivative of the Leaky ReLU function for X
        """
        return np.where(x >= 0, 1, 0.01)

    def dropout(self, x):
        """
        This function applies dropout to the hidden layer
        :param x: the hidden layer outputs before applying the dropout
        :return: the hidden layer with dropout
        """
        # create a mask for the dropout, the mask size is the same as the hidden layer size
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        # if the mask is 1, keep the value, else, set the value to 0
        return x * mask

    def forward(self, x, training=False):
        """
        This function calculates the model prediction for X using the activation function
        and the softmax function.
        :param x: the input matrix
        :param training: whether to apply dropout to the hidden layer
        :return: the model prediction for X
        """
        z1 = x @ self.w1 + self.b1
        h = self.activation_function(z1)
        # Apply dropout only during training
        if training:
            h = self.dropout(h)

        z2 = h @ self.w2 + self.b2
        return softmax(z2)

    def loss(self, y_pred, y):
        """
        This function calculates the loss of the model
        :param y_pred: the output of the model
        :param y: the labels
        :return: the loss of the model
        """
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0] + self.mu * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))

    def backward(self, x, y, y_pred):
        """
        This function calculates the gradients of the model and updates the model parameters
        :param x: the input data
        :param y: the labels
        :param y_pred: the output of the model
        :return: None
        """
        self.t += 1
        z1 = x @ self.w1 + self.b1
        h = self.activation_function(z1)

        # Calculate the gradients
        self.w2_grad = (h.T @ (y_pred - y)) / y.shape[0] + 2 * self.mu * self.w2
        self.b2_grad = np.sum(y_pred - y, axis=0) / y.shape[0]

        w2_delta = (y_pred - y) @ self.w2.T * self.activation_derivative(z1)
        self.w1_grad = (x.T @ w2_delta) / y.shape[0] + 2 * self.mu * self.w1
        self.b1_grad = np.sum(w2_delta, axis=0) / y.shape[0]

        # Apply momentum to the gradients if used
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

        # Update parameters
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

        # Update previous gradients for momentum if used
        if self.momentum is not None:
            self.prev_w1_grad = self.w1_grad.copy()
            self.prev_b1_grad = self.b1_grad.copy()
            self.prev_w2_grad = self.w2_grad.copy()
            self.prev_b2_grad = self.b2_grad.copy()

    def update_lr(self):
        """
        This function updates the learning rate of the model using the learning rate decay
        :return: None
        """
        self.lr *= self.lr_decay

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def train(model, x_train, y_train, x_val, y_val, epochs=1000, batch_size=100, verbose=100,
          batch_size_add=0, max_add=25, random_shift=1):
    """
    This function trains the model using the training data and validates it using the validation data
    :param model: the model to train
    :param x_train: the training data
    :param y_train: the training labels
    :param x_val:  the validation data
    :param y_val: the validation labels
    :param epochs: the number of epochs to train the model
    :param batch_size: the batch size
    :param verbose: the number of epochs to print the loss and accuracy
    :param batch_size_add: the number to add to the batch size after each epoch
    :param max_add: the maximum number to add to the pixels (uniformly)
    :param random_shift: the maximum number to shift the image (vertically and horizontally)
    :return: the best model, the training loss history, the validation loss history, the training accuracy history,
    """
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_model = None
    best_val_acc = 0

    # iterate over the epochs
    for epoch in range(epochs):
        batch_loss_arr = []
        batch_acc_arr = []
        # iterate over the batches
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            # augment the data
            x_batch = data_augmentation(x_batch, y_batch, max_add, random_shift)
            # forward pass
            y_pred = model(x_batch, training=True)
            y_pred_class = np.argmax(y_pred, axis=1)
            y_batch_class = np.argmax(y_batch, axis=1)
            batch_acc_arr.append(np.mean(y_pred_class == y_batch_class))
            batch_loss_arr.append(model.loss(y_pred, y_batch))
            # backward pass
            model.backward(x_batch, y_batch, y_pred)

        # calculate the training loss and accuracy and the validation loss and accuracy after each epoch
        train_loss = np.mean(batch_loss_arr)
        train_loss_history.append(train_loss)
        train_acc = np.mean(batch_acc_arr)
        train_acc_history.append(train_acc)

        y_val_pred = model(x_val)
        val_loss = model.loss(y_val_pred, y_val)
        val_loss_history.append(val_loss)
        y_val_pred_class = np.argmax(y_val_pred, axis=1)
        val_acc = np.mean(y_val_pred_class == np.argmax(y_val, axis=1))
        val_acc_history.append(val_acc)

        # save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

        # print the loss and accuracy after each verbose epochs
        if epoch % verbose == 0:
            print('epoch {}, train loss {}, val loss {}, train acc {}, val acc {}'.format(epoch, train_loss,
                                                                                          val_loss, train_acc, val_acc))
        # update the learning rate after each epoch and the batch size if needed
        model.update_lr()
        batch_size += batch_size_add

    return best_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


if __name__ == '__main__':
    # load the data
    x_train, x_val, x_test, y_train, y_val = load_data()
    # visualize the data
    visualize(x_train, y_train, h=20)

    # data normalization
    mean_train = np.mean(x_train, axis=0)
    std_train = np.std(x_train, axis=0)

    data_normalization_type = ''

    # normalize the data using z-score normalization or min-max normalization
    if data_normalization_type == 'z_score':
        x_train = (x_train - mean_train) / (std_train + eps)
        x_val = (x_val - mean_train) / (std_train + eps)
        x_test = (x_test - mean_train) / (std_train + eps)
    else:
        x_train = x_train / 255
        x_val = x_val / 255
        x_test = x_test / 255


    # shuffle the data and one hot encode the labels
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    y_train_hot = one_hot_encoding(y_train)
    y_val_hot = one_hot_encoding(y_val)

    # define the hyperparameters
    lr = 0.001
    lr_decay = 0.99
    epochs = 201
    batch_size = 128
    mu = 0.0001
    verbose = 1
    hidden_size = 128
    momentum = None
    batch_size_add = 1
    factor_init = 4
    max_add = 25
    random_shift = 1
    dropout_rate = 0

    # define the models and train them
    # model = LogisticRegression(mu=mu, lr=lr, lr_decay=lr_decay, factor_init=factor_init)
    model = FashionNet(mu=mu, lr=lr, hidden_size=hidden_size, activation_function='relu', momentum=momentum,
                       lr_decay=lr_decay, factor_init=factor_init, beta1=0.9, beta2=0.999, dropout_rate=dropout_rate)

    # start the training and measure the time it takes
    t1 = time()
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = (
        train(model, x_train, y_train_hot,
                                        x_val, y_val_hot,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose=verbose,
                                        batch_size_add=batch_size_add,
                                        max_add=max_add,
                                        random_shift=random_shift))
    # save the model
    np.save('model2.npy', model)

    # print the training time
    print('Training time: ', time() - t1)

    # plot the loss and accuracy history
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


    # Step 1: Compute Accuracy for Each Class
    y_test_pred = model(x_test)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)

    y_pred = model(x_val)
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

    # Step 4: Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=dict_class_name.values(),
                yticklabels=dict_class_name.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Lets show the wrong samples of class 6

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
