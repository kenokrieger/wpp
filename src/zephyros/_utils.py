from sklearn.preprocessing import StandardScaler


def scale(x, y):
    """
    Scale the data for the learning process and return the scaled
    data and the scalers used in the process.

    Args:
        x (np.ndarray): The unscaled learning data containing the feature
            values.
        y (np.ndarray): The unscaled learning data containing the target values.

    Returns:
        tuple: The scalers and the scaled values.

    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(x)
    x_scaled = feature_scaler.transform(x)
    y = y.reshape(-1, 1)
    target_scaler.fit(y)
    y_scaled = target_scaler.transform(y)
    return (feature_scaler, target_scaler), (x_scaled, y_scaled)


def sample_and_scale(x, features, target, test_percentage, random_state):
    """
    Sample and scale the data for the learning process and return the scaled
    data and the scalers used in the process.

    Args:
        x (pandas.DataFrame): The unsampled learning data.
        features (list): Column names of the features to use in the learning
            process.
        target (list): Column name of the target for the learning process.
        test_percentage (float): Percentage of data to use for testing.
        random_state (np.random.Generator or None): The random state to use for
            the sampling.

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
