from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve
from sklearn.svm import SVC
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

    svm_model = SVC(kernel="linear").fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print('Accuracy:', accuracy_score(y_pred, y_test))
    # print('Rapor:', classification_report(y_pred, y_test))

    params = {
        "C": np.arange(1,100)
    }

    # from sklearn.model_selection import GridSearchCV

    # svc_cv = GridSearchCV(svm_model, params, cv=10, verbose=2, n_jobs=-1)
    # svc_cv.fit(X_train, y_train)
    # print(svc_cv.best_params_)

    svm_tuned_model = SVC(kernel="linear", C=1).fit(X_train, y_train)
    y_tuned_pred = svm_tuned_model.predict(X_test)

    print('Tuned Accuracy:', accuracy_score(y_tuned_pred, y_test))
    # print('Tuned Rapor:', classification_report(y_tuned_pred, y_test))

