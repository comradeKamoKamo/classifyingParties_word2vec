#%%
from pathlib import Path
import copy
import pickle
from logging import getLogger, INFO, StreamHandler

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dropout,Dense,Reshape
from keras import utils
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import precision_recall_curve,auc,roc_curve,confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

class Train:

    def get_politicians(self):
        data_dict = self.load_data()
        return list(data_dict.keys())

    def main(self,extext=""):
        data_dict = self.load_data()
        X, y, politicians = self.data_to_xy(data_dict)
        n_classes = len(politicians)
        #X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.3)

        with Path("split_data.pickle").open("rb") as f:
            split_info = pickle.load(f)
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for xi, yi in zip(X,y):
            if xi[0] in split_info["test"][xi[1]]:
                X_test.append(xi[2])
                y_test.append(yi)
            else:
                X_train.append(xi[2])
                y_train.append(yi)
        
        X_train = np.array(X_train,dtype=float)
        y_train = np.array(y_train,dtype=float)
        X_test = np.array(X_test,dtype=float)
        y_test = np.array(y_test,dtype=float)
        print(X_test)

        y_test_raw = copy.copy(y_test)
        y_train = utils.np_utils.to_categorical(y_train, n_classes)
        y_test = utils.np_utils.to_categorical(y_test, n_classes)
        model = self.build_model(n_classes)
        self.train(X_train,y_train,X_test,y_test,model,extext)
        self.test(model,X_test,y_test_raw,politicians,extext)

    def build_model(self,n_classes):
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

    def train(self,X_train,y_train,X_test,y_test,model,extext=""):
        # 目的関数にSGD、誤差関数にloglessを用いる
        model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        es_cb = EarlyStopping()
        model.fit(X_train,y_train,
            validation_data=(X_test,y_test),epochs=1024,callbacks=[es_cb])
        model.save_weights("model{0}.hdf5".format(extext))
        r = model.evaluate(X_test,y_test)
        print(r)
        with Path("data/accuracy.txt").open("a") as f:
            f.write(str(r[1]) + "\n")
        return model

    def test(self,model,X_test,y_test,politicians,extext=""):
        c_acc = 0
        n_classes = len(politicians)

        raw_preds = model.predict(X_test)
        preds = []
        for raw_pred in raw_preds:
            preds.append(np.where(raw_pred == max(raw_pred))[0][0])
        y_test = np.array(y_test)
        preds = np.array(preds)

        cm = confusion_matrix(y_test,preds)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        np.save("data/cm{0}.npy".format(extext),cm)
        
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
        for i in range(n_classes):
            labels.append(i)
        y_test = label_binarize(y_test,classes=labels)
        preds = label_binarize(preds,classes=labels)
        precision, recall , _ = precision_recall_curve(y_test.ravel(),preds.ravel())
        prc_auc = auc(recall,precision)
        np.save("data/precision{0}.npy".format(extext),precision)
        np.save("data/recall{0}.npy".format(extext),recall)

        
        plt.figure()
        plt.step(recall, precision, color="b", alpha=0.2,where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("P/R (Micro Average) AUC={0}".format(prc_auc))
        plt.show()
        

        fpr , tpr , _ = roc_curve(y_test.ravel(),preds.ravel())
        roc_auc = auc(fpr,tpr)
        np.save("data/fpr{0}.npy".format(extext),fpr)
        np.save("data/tpr{0}.npy".format(extext),tpr)

        
        plt.figure()
        plt.step(fpr, tpr, color="r", alpha=0.2,where="post")
        plt.fill_between(fpr,tpr, step="post", alpha=0.2, color="r")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([0,1],[0,1],linestyle="dashed",color="pink")
        plt.title("ROC (Micro Average) AUC={0}".format(roc_auc))
        plt.show()
        

        return c_acc 

    def load_data(self):
        data = dict()
        for npy in Path("politicians").glob("*.pickle"):
            name = npy.stem
            with npy.open("rb") as f:
                data[name] = pickle.load(f)
        return data

    def data_to_xy(self,dict_data):
        politicians = []
        x, y = [], []
        i = 0
        for key , data in zip(dict_data.keys(), dict_data.values()) :
            politicians.append(key)
            for vector in data:
                x.append(vector)
                y.append(i)
            i += 1
        return x, np.asarray(y), politicians

if __name__ == "__main__" :
    train = Train()
    train.main()