from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.ensemble import RandomForestClassifier


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

    positiveIndex = []
    negativeIndex = []
    for i, train in enumerate(y):
        if train == 1:
            positiveIndex.append(i)
        if train == 0:
            negativeIndex.append(i)

    positiveCases = X[positiveIndex]
    negativeCases = X[negativeIndex]

    positiveLabels = y[positiveIndex]
    negativeLabels = y[negativeIndex]


    #test   
    n = 14
    index = np.random.choice(positiveCases.shape[0], n, replace=False)
    X_test_positive = positiveCases[index]
    y_test_positive = positiveLabels[index]

    n = 9
    index = np.random.choice(negativeCases.shape[0], n, replace=False)
    X_test_negative = negativeCases[index]
    y_test_negative = negativeLabels[index]


    X_test = np.concatenate((X_test_positive, X_test_negative), axis=0)
    y_test = np.concatenate((y_test_positive, y_test_negative), axis=0)

    
    #train
    n = 59
    index = np.random.choice(positiveCases.shape[0], n, replace=False)
    X_train_positive = positiveCases[index]
    y_train_positive = positiveLabels[index]

    n = 39
    index = np.random.choice(negativeCases.shape[0], n, replace=False)
    X_train_negative = negativeCases[index]
    y_train_negative = negativeLabels[index]


    X_train = np.concatenate((X_train_positive, X_train_negative), axis=0)
    y_train = np.concatenate((y_train_positive, y_train_negative), axis=0)

    return X_train, X_test, y_train, y_test

def statistics(type, pred, real, index):
    test_accuracy = accuracy_score(pred, real)
    print("Accuracy--Index => \n" + str(index + 1) , type, test_accuracy)
    test_error = np.sqrt(mean_squared_error(pred, real))
    print("Error--Index => \n" + str(index + 1), type, test_error)

    Raports['accuricies'].append([test_accuracy.tolist(), str(index + 1), str(type)])
    Raports['errors'].append([test_error.tolist(), str(index + 1), str(type)])
    print('------------------------------------------------------------------')
    try:
        auc_score = roc_auc_score(real,pred)
        print("AUC Score--Index => \n" + str(index + 1), type, auc_score)
        Raports['aucScores'].append([auc_score.tolist(), str(index + 1), str(type)])
    except ValueError:
        pass

    print('------------------------------------------------------------------')
    confusion_matrix_results = confusion_matrix(real, pred)
    print("Train Confusion Matrix--Index => \n"+ str(index + 1), type)
    Raports['counfusionMatrix'].append([confusion_matrix_results.tolist(), str(index + 1), str(type)])
    print(confusion_matrix_results, str(index + 1), str(type))

    print('------------------------------------------------------------------')
    report = classification_report(real, pred, target_names=["covid", "not covid"])
    print("Report--Index => \n" + str(index + 1), type, report)
    Raports['raports'].append([report, str(index + 1), str(type)])
    print('------------------------------------------------------------------')

if __name__ == "__main__":

    histories = []
    rocsTrain = []
    rocsValidation = []
    rocsTest = []
    precisionsTrain = []
    precisionsValidation = []
    precisionsTest = []

    Raports = {
        "raports": [],
        "counfusionMatrix": [],
        "aucScores": [],
        "accuricies": [],
        "errors": []
    }

    for i in range(5):
        X_train, X_test, y_train, y_test = prepare_datasets(0.2)


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
        y_pred_train = rf_tuned.predict(X_train)
        
        statistics('Train', y_train,y_pred_train, i)
        statistics('Test', y_test,y_tuned_pred, i)

        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred)

        rocsTrain.append([fpr_train, tpr_train])
        rocsTest.append([fpr_test, tpr_test])



        train_precision, train_recall, _ = precision_recall_curve(y_train, y_pred_train)
        test_precision, test_recall, _ = precision_recall_curve(y_test, y_pred)

        precisionsTrain.append([train_precision, train_recall])
        precisionsTest.append([test_precision, test_recall])
    
    fig, axs = plt.subplots(5)
    for j, roc in enumerate(rocsTrain):
        # create train
        axs[j].plot(roc[0], roc[1], linestyle='--', label='Train' + str(j+1))
        axs[j].plot(rocsTest[j][0], rocsTest[j][1], linestyle='--',label="Test" + str(j+1))
        axs[j].set_ylabel("Roc")
        axs[j].legend(loc="lower right")
        axs[j].set_title("Roc")
        
    plt.show()

    fig, axs = plt.subplots(5)
    for k, precisionRecall in enumerate(precisionsTrain):
        # create train
        axs[k].plot(precisionRecall[0], precisionRecall[1], linestyle='--', label='Train' + str(k+1))
        axs[k].plot(precisionsTest[k][0], precisionsTest[k][1], linestyle='--',label="Test" + str(k+1))
        axs[k].set_ylabel("Precision Recall")
        axs[k].legend(loc="lower right")
        axs[k].set_title("Precision Recall")
        
    plt.show()

    with open("./raportsRFC.json", "w") as fp:
        json.dump(Raports, fp, indent=4)

    