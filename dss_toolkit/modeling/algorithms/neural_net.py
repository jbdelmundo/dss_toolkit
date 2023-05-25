from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def train_nn(train_data, train_labels,test_data=None, test_targets=None, **kwargs):

    input_shape = (train_data.shape[1],)
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.05))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="rmsprop", loss="mse")

    model.fit(
        train_data,
        train_labels,
        epochs=kwargs.get("nn_epochs", 100),
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20)],
    )
    return model


def predict_nn(model, X, y):
    proba = model.predict(X).reshape(-1, 1).flatten()
    return proba
