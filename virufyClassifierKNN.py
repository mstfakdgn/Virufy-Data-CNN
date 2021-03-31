from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
import json

DATA_PATH="dataVirufy.json"

def load_data(data_path):
    """Loads training dataset from json file

    :param data_path(str): path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))

    return X,y

def prepare_datasets(test_size):

    # laod data
    X, y = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_datasets(0.2)

    knn = KNeighborsClassifier()
    knn_model = knn.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    print('Accuracy:', accuracy_score(y_pred, y_test))
    # print('Rapor:', classification_report(y_pred, y_test))

    # knn_params = {
    #     "n_neighbors": np.arange(1, 50)
    # }

    # from sklearn.model_selection import GridSearchCV

    # knn = KNeighborsClassifier()
    # knn_cv = GridSearchCV(knn, knn_params, cv=10)
    # knn_cv.fit(X_train, y_train)

    # print("Best score:" + str(knn_cv.best_score_))
    # print("Best parameters:" + str(knn_cv.best_params_))


    knn = KNeighborsClassifier(n_neighbors=1)
    knn_tuned = knn.fit(X_train, y_train)
    y_pred = knn_tuned.predict(X_test)
    print('Tuned Accuracy:', accuracy_score(y_pred, y_test))
    # print('Tuned Rapor:', classification_report(y_pred, y_test))