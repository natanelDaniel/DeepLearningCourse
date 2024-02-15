import numpy as np
from Ex2 import load_data
# In this file we will load the models and predict the test set using the models
# and save the predictions in csv files
if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val = load_data()
    # Load the model
    model = np.load('model1.npy', allow_pickle=True).item()
    # Predict the test set
    y_pred = model(x_test)
    # from hot encoding to label
    y_pred = np.argmax(y_pred, axis=1)
    # save the prediction in csv file, each row is the prediction for one test sample
    np.savetxt('lr_pred.csv', y_pred, delimiter=',', fmt='%d')

    model = np.load('model2.npy', allow_pickle=True).item()
    # Predict the test set
    y_pred = model(x_test)
    # from hot encoding to label
    y_pred = np.argmax(y_pred, axis=1)
    np.savetxt('NN_pred.csv', y_pred, delimiter=',', fmt='%d')