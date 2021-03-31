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

    from sklearn.ensemble import RandomForestClassifier

    rfm = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rfm.predict(X_test)
    print('Accuracy:' , accuracy_score(y_pred, y_test))

    rf_params = {
        'max_depth': [2,5,8,10,20],
        'max_features': [2,5,8,16],
        'n_estimators': [10,500,1000,1500],
        'min_samples_split' : [2,5,10,20]
    }

    # from sklearn.model_selection import GridSearchCV

    # # n_jobs = -1 parameter is to much full performance
    # rf_cv_model = GridSearchCV(rfm, rf_params, cv=10, verbose=2, n_jobs=-1)
    # rf_cv_model.fit(X_train, y_train)
    # print(rf_cv_model.best_params_)

    rf_tuned = RandomForestClassifier(max_depth=10, max_features=16, n_estimators=10, min_samples_split=5)
    rf_tuned.fit(X_train, y_train)
    y_tuned_pred = rf_tuned.predict(X_test)
    print('Tuned Score:', accuracy_score(y_test, y_tuned_pred))

    