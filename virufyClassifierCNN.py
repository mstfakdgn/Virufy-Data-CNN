import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import roc_curve, auc,roc_auc_score


DATA_PATH="dataVirufy.json"

def load_data(data_path):
    """Loads training dataset from json file

    :param data_path(str): path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    mfccsArray = []
    for mfcc in data['MFCCs']:
        print(np.array(mfcc).shape)
        mfccsArray.append(mfcc)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    pdb.set_trace()
    return X,y

def prepare_datasets(test_size, validation_size):

    # laod data
    X, y = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create mdoel
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (5,5), activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(1,1), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(1,1), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(128, (2,2), activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(1,1), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))

    # 4rd conv layer
    model.add(keras.layers.Conv2D(256, (2,2), activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(1,1), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))

    # 5rd conv layer
    model.add(keras.layers.Conv2D(512, (2,2), activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
        # max pooling
    model.add(keras.layers.MaxPool2D((2,2), strides=(1,1), padding="same"))
        # standartizes current layer, speed up training
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.1))


    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    # model.add(keras.layers.Dropout(0.1))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X,y):

    X = X[np.newaxis, ...]

    y_pred = model.predict(X) # X -> 3d array (?1?, 130, 13, 1)

    #y_pred 2d array -> [[0.1,0.2,..., 0.3]]
    # extract index with max value
    predicted_index = np.argmax(y_pred, axis=1) # [3]

    print("Expected index: {}, Predicted index: {} ". format(y, predicted_index))

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    
    # create train, validation and test sets # test_size, validation_size
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.2, 0.2)
    
    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train the CNN #Epocs 300
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Accuracy Test {}'.format(test_accuracy))
    print('Error Test {}'.format(test_error))
    
    # make prediction on a sample
    X = X_test[10]
    y = y_test[10]

    predict(model, X,y)
    #plot the accuracy and error over the epochs
    plot_history(history)

    y_pred_all=model.predict(X_test)
    y_pred_index = np.array([])
    for pred in y_pred_all:
        predicted_index = np.argmax(pred, axis=0)
        y_pred_index = np.append(y_pred_index, predicted_index)
        
    auc_score=roc_auc_score(y_test,y_pred_index)
    print('AUC:', auc_score)
