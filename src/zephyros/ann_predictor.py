import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


def learn_and_predict(learn_data, predict_data, features, target,
                      test_percentage=0.33, xvalidate=None, seed=None):
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


def learn(x, features, target, test_percentage=0.33, random_state=None):
    """
    Args:
        x(pandas.DataFrame): The data to use for learning a model.
        features(list): The features to use in the learning process.
        target(list): The target(s) for the learning process.
        test_percentage(float): Percentage of *x* to use for testing.
        random_state(np.random.Generator): Random generator for the sampling.

    Returns:
        keras.Models.Sequential: The learned model.

    """
    test = x.sample(frac=test_percentage, random_state=random_state)
    # complement of the sampled data
    train = x.iloc[x.index.difference(test.index)]
    x_train = train[features].to_numpy()
    y_train = train[target].to_numpy()
    # unused data from the learning set
    x_test = test[features].to_numpy()
    y_test = test[target].to_numpy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(x_train)
    x_scaled = feature_scaler.transform(x_train)

    y_train = y_train.reshape(-1, 1)
    target_scaler.fit(y_train)
    y_scaled = target_scaler.transform(y_train)

    model = Sequential()
    model.add(Dense(units=5, input_dim=2, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_scaled, y_scaled, batch_size=200, epochs=100, verbose=100)
    return model, (feature_scaler, target_scaler)


def predict(model, scaler, x):
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(xgboost.XGBRegressor): The learned model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.array: The predicted values for the target learned.

    """
    x_scaled = scaler[0].transform(x)
    y = model.predict(x_scaled)
    return scaler[1].inverse_transform(y).ravel()
