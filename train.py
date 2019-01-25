#%%
from pathlib import Path
import copy
from logging import getLogger, INFO, StreamHandler

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dropout,Dense,Reshape
from keras import utils
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,auc,roc_curve,confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

#%%
def main():
    data_dict = load_data()
    X, y, politicians = data_to_xy(data_dict)
    n_classes = len(politicians)
    y = utils.np_utils.to_categorical(y,n_classes)
    X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.3)
    model = build_model(n_classes)
    train(X_train,y_train,X_test,y_test,model)
    test(model,X_test,y_test,politicians)
#%%
def build_model(n_classes):
    model = Sequential()
    model.add(Dense(128, input_shape=(200,), activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(n_classes,activation="softmax"))    
    model.summary()
    json_string = model.to_json()
    Path("model.json").open("w").write(json_string)
    return model

def train(X_train,y_train,X_test,y_test,model):
    # 目的関数にSGD、誤差関数にloglessを用いる
    model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    es_cb = EarlyStopping()
    model.fit(X_train,y_train,
        validation_data=(X_test,y_test),epochs=256,callbacks=[es_cb])
    model.save_weights("model.hdf5")
    r = model.evaluate(X_test,y_test)
    print(r)
    return model

def test(model,X_test,y_test,politicians):
    c_acc = 0
    b_acc = 0
    n_classes = len(politicians)

    raw_preds = model.predict(X_test)
    preds = []
    for raw_pred in raw_preds:
        preds.append(np.where(raw_pred == max(raw_pred))[0][0])
    y_test = np.array(y_test)
    preds = np.array(preds)

    cm = confusion_matrix(y_test,preds)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    np.save("cm.npy",cm)
    
    fig = plt.figure()
    ax = plt.subplot()
    cax = ax.matshow(cm, interpolation="nearest", cmap="autumn_r")
    fig.colorbar(cax)
    ax.set_xticklabels([""]+politicians)
    ax.set_yticklabels([""]+politicians)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()

    labels = []
    for i in range(len(parties)):
        labels.append(i)
    y_test = label_binarize(y_test,classes=labels)
    preds = label_binarize(preds,classes=labels)
    precision, recall , _ = precision_recall_curve(y_test.ravel(),preds.ravel())
    prc_auc = auc(recall,precision)
    np.save("precision.npy",precision)
    np.save("recall.npy",recall)

    
    plt.figure()
    plt.step(recall, precision, color="b", alpha=0.2,where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P/R (Micro Average) AUC={0}".format(prc_auc))
    plt.show()
    

    fpr , tpr , _ = roc_curve(y_test.ravel(),preds.ravel())
    roc_auc = auc(fpr,tpr)
    np.save("fpr.npy",fpr)
    np.save("tpr.npy",tpr)

    
    plt.figure()
    plt.step(fpr, tpr, color="r", alpha=0.2,where="post")
    plt.fill_between(fpr,tpr, step="post", alpha=0.2, color="r")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0,1],[0,1],linestyle="dashed",color="pink")
    plt.title("ROC (Micro Average) AUC={0}".format(roc_auc))
    plt.show()
    

    return c_acc 

#%%
def load_data():
    data = dict()
    for npy in Path("Data").glob("*.npy"):
        name = npy.stem
        data[name] = np.load(str(npy))
    return data

def data_to_xy(dict_data):
    politicians = []
    x, y = [], []
    i = 0
    for key , data in zip(dict_data.keys(), dict_data.values()) :
        politicians.append(key)
        for vector in data:
            x.append(vector)
            y.append(i)
        i += 1
    return np.asarray(x), np.asarray(y), politicians

#%%
if __name__ == "__main__" :
    main()    