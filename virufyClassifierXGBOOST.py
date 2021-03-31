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

    from xgboost import XGBClassifier

    xgb_model = XGBClassifier().fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print('Accuracy:', accuracy_score(y_pred, y_test))
    print('Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


    # #Tuning
    # xgb_grid = {
    #     "n_estimators" : [50,100,500,1000],
    #     "subsample" : [0.2,0.4,0.6,0.8,1.0],
    #     "max_depth" : [3,4,5,6,7,8],
    #     "learning_rate" : [0.1, 0.01, 0.001, 0.0001],
    #     "min_samples_split" : [2,5,10,12]
    # }

    # from sklearn.model_selection import GridSearchCV

    # xgb_cv_model = GridSearchCV(xgb_model, xgb_grid, cv=5, n_jobs=-1, verbose=2)
    # xgb_cv_model.fit(X_train,y_train)
    # print(xgb_cv_model.best_params_)


    xgb_tuned = XGBClassifier(learning_rate=0.1, max_dept=3, min_samples_split=2, n_estimators=500, subsample=0.4).fit(X_train,y_train)
    y_tuned_pred = xgb_tuned.predict(X_test)
    print('Tuned Accuracy:', accuracy_score(y_test, y_tuned_pred))
    print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))