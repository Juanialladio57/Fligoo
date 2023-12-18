#%%
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#%%

def features_and_target(df, features, target):
    """
    Prepares feature matrix and target vector for machine learning models by encoding 
    categorical variables using One-Hot Encoding and scaling numerical features.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame containing the dataset.
    features : list of str
        List of column names to be used as features.
    target : str
        The name of the target variable.

    Returns
    -------
    X_final : numpy.ndarray
        The final feature matrix after encoding categorical variables and scaling numerical variables.
    y : numpy.ndarray
        The target vector.

    """
    # Identify categorical and numerical variables
    categorical_vars = df[features].select_dtypes(include='object').columns
    numeric_vars = df[features].select_dtypes(include=['int64', 'float64']).columns

    # Encode categorical variables using One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    # Transform categorical features into a 2D numeric array
    X_encoded = encoder.fit_transform(df[categorical_vars])

    # Scale numerical features
    scaler = StandardScaler()
    # Standardize numerical features by removing the mean and scaling to unit variance
    X_scaled = scaler.fit_transform(df[numeric_vars])

    # Combine encoded and scaled features
    # Concatenate encoded categorical features and scaled numerical features
    X_final = np.concatenate((X_encoded, X_scaled), axis=1)

    # Prepare the target variable
    # Extract the target variable as a numpy array
    y = df[target].values

    return X_final, y
