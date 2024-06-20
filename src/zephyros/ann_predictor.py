import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


def learn_and_predict(learn_data, predict_data, features, target,
                      test_percentage=0.33, xvalidate=None, seed=None,
                      config=None):
    """
    Convenience function combining ann_predictor.learn and ann_predictor.predict
    into a single function. Given feature values and a target to predict, learn
    a model and use it to predict the target variable in the *predict_data*.

    Args:
        learn_data(pandas.DataFrame): The data to use for learning the model.
        predict_data(pandas.DataFrame): The data to use for the prediction.
        features(list): The features to use for learning and predicting.
        target(list): The target(s) to predict.
        test_percentage(float): Percentage of *learn_data* to use for testing.
        xvalidate(int or None): Choose the number of cross validations.
            Defaults to None.
        seed(int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        config (dict): A configuration for the structure of the neural network
            and the learning process.

    Returns:
        np.ndarray: The predicted values.

    """
    random_state = np.random.default_rng(seed=seed)
    if not xvalidate:
        model, scaler = learn(learn_data, features, target,
                              test_percentage=test_percentage,
                              random_state=random_state)
        x_pred = predict_data[features].to_numpy()
        return predict(model, scaler, x_pred)

    nrows = predict_data.shape[0]
    predicted = np.empty((xvalidate, nrows))

    for i in range(xvalidate):
        model, scaler = learn(learn_data, features, target,
                              test_percentage=test_percentage,
                              random_state=random_state)
        x_pred = predict_data[features].to_numpy()
        predicted[i] = predict(model, scaler, x_pred)
    return predicted


def learn(x, features, target, test_percentage=0.33, random_state=None,
          config=None):
    """
    Args:
        x(pandas.DataFrame): The data to use for learning a model.
        features(list): The features to use in the learning process.
        target(list): The target(s) for the learning process.
        test_percentage(float): Percentage of *x* to use for testing.
        random_state(np.random.Generator): Random generator for the sampling.
        config (dict): A configuration for the structure of the neural network
            and the learning process.

    Returns:
        tuple: The learned model and the scalers.

    """
    cb_cls = {"EarlyStopping": EarlyStopping}
    if config is None:
        config = {
            "layers": [
                {"units": 50, "kernel_initializer": "normal", "activation": "relu"},
                {"units": 20, "kernel_initializer": "normal", "activation": "relu"},
                {"units": 10, "kernel_initializer": "normal", "activation": "relu"},
                {"units": 3, "kernel_initializer": "normal", "activation": "tanh"},
                {"units": 1, "kernel_initializer": "normal"},
            ],
            "compile": {"loss": "mean_squared_error", "optimizer": "adam"},
            "callbacks": {"EarlyStopping": {"monitor": "val_loss", "patience": 5}},
            "options": {"batch_size": 200, "epochs": 1_000, "verbose": 1}
        }
    scaler, values = _sample_and_scale(x, features, target,
                                       test_percentage, random_state)

    model = Sequential([Dense(**c) for c in config["layers"]])
    model.compile(**config["compile"])

    cb_opt = config["callbacks"]
    callbacks = [cb_cls[cb](**opt) for cb, opt in cb_opt.items() if cb in cb_cls]
    model.fit(values[0], values[1],
              validation_data=values[2:],
              callbacks=callbacks,
              **config["options"])
    return model, scaler


def predict(model, scaler, x):
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(xgboost.XGBRegressor): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.array: The predicted values for the target learned.

    """
    x_scaled = scaler[0].transform(x)
    y = model.predict(x_scaled)
    return scaler[1].inverse_transform(y).ravel()


def _sample_and_scale(x, features, target, test_percentage, random_state):
    """
    Sample and scale the data for the learning process and return the scaled
    data and the scalers used in the process.

    Args:
        x (pandas.DataFrame): The unsampled learning data.
        features (list): Column names of the features to use in the learning
            process.
        target (list): Column name of the target for the learning process.
        test_percentage (float): Percentage of data to use for testing.
        random_state (int or None): The random state to use for the sampling.

    Returns:
        tuple: The scalers and the scaled values.

    """
    test = x.sample(frac=test_percentage, random_state=random_state)
    # complement of the sampled data
    train = x.iloc[x.index.difference(test.index)]
    x_train = train[features].to_numpy()
    y_train = train[target].to_numpy()
    x_test = test[features].to_numpy()
    y_test = test[target].to_numpy()
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(x_train)
    x_scaled = feature_scaler.transform(x_train)
    x_test_scaled = feature_scaler.transform(x_test)
    y_train = y_train.reshape(-1, 1)
    target_scaler.fit(y_train)
    y_scaled = target_scaler.transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    return ((feature_scaler, target_scaler),
            (x_scaled, y_scaled, x_test_scaled, y_test_scaled))
