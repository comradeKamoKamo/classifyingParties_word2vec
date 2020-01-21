from pathlib import Path
import numpy as np 
from keras import utils
from keras.models import Input, Model
from keras.layers import Dense, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def load_data(npy_path, train_rate, class_id):
    data = np.load(npy_path)
    l = len(data)
    rands = [np.random.rand() for i in range(l)]
    X_train = [data[i] for i in range(l) if rands[i] <= train_rate]
    X_test = [data[i] for i in range(l) if rands[i] > train_rate]
    y_train = [class_id for i in X_train]
    y_test = [class_id for i in X_test]
    return X_train, X_test, y_train, y_test

def load_data_set(dir_path, seed, train_rate):
    X_train, X_test, y_train, y_test = [], [], [], []
    labels = []
    np.random.seed(seed)
    for i, p in enumerate(Path(dir_path).glob("*.npy")):
        xtr, xte, ytr, yte = load_data(str(p), 0.7, i)
        X_train.extend(xtr)
        X_test.extend(xte)
        y_train.extend(ytr)
        y_test.extend(yte)
        labels.append(p.name)
    y_train = utils.np_utils.to_categorical(y_train)
    y_test = utils.np_utils.to_categorical(y_test)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), labels

def create_model(train_shape, n_classes):
    input_tensor = Input(train_shape)
    Bi_LSTM = Bidirectional(LSTM(356))(input_tensor)
    output_tensor = Dense(n_classes, activation='softmax')(Bi_LSTM)

    model = Model(input_tensor, output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['mae', 'mse', 'acc'])

    return model

def train(model, X_train, X_test, y_train, y_test):
    model.summary()
    model.fit(X_train,
            y_train,
            epochs = 100,
            batch_size = 128,
            validation_data=(X_test, y_test),
            shuffle=False,
            verbose = 1,
            callbacks = [
                EarlyStopping(patience=5, monitor='val_acc', mode='max'),
                ModelCheckpoint(monitor='val_acc', mode='max', filepath="model.hdf5", save_best_only=True)
            ])
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, labels = load_data_set("features/", 114514, 0.7)
    model = create_model(X_train[0].shape, len(labels))
    train(model, X_train, X_test, y_train, y_test)
