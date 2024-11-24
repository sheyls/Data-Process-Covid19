"""
This file should contain the scalings and other transformations that are parametric and require using only the training set.

BEFORE SUNDAY!?

## EDA
- [x] Analizar el balance del target


## Preproc
- [X] Separar en train-test
- [X] transformaciones para ponernos mas normales (log)
- [Done for now] outliers (detect and remove)
- [Sheyla] missings (fill nan's)
    - [ ] Save the imputer fitted with the training data as a pickle file to fill the validation later
    - [ ] Impute the nominal variables
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(mixed_dataset)
- [x] transformar pais a numerico (vector)
- [x] Balancear las clases < 15 - 85 por ahora

## Entrenar
Implementar modelos:
- [ ] Dec trees (print  el arbol)
- [ ] Rule Ind (print las reglas)
- [ ] Log reg (pintar boundary with pairs)
- [ ] SVM (pintar boundary with pairs)
- [ ] Naive Bayes (net dependency)
- [ ] Random forest (pintar unos cuantos arboles)



"""
import pickle

import numpy as np
from pandas.core.common import random_state
from scipy.special import ellip_harm
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
# from spacy.cli import train
# from sqlalchemy.dialects.mssql.information_schema import columns
import matplotlib.pyplot as plt
import seaborn as sns


from Assignment2.eda import multibar_plots, save_boxplots_and_histograms

ROOT = "./COVID19_data/"
DS_NAME = "extended_df.csv"

def balance_classes(df, target_column, output_file_name="class_balance_results.txt"):
    """
    Balances the classes in a dataset using the specified method.

    :param df: pandas DataFrame containing the dataset.
    :param target_column: The name of the target column to balance.
    :param method: The balancing method to use oversampling using SMOTE.
    :return: A new balanced DataFrame.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    balancer = SMOTE(random_state=15)

    X_balanced, y_balanced = balancer.fit_resample(X, y)
    
    # Combine back into a DataFrame
    balanced_df = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), 
                             pd.DataFrame(y_balanced, columns=[target_column])], axis=1)
    
    # Save results to a text file
    with open(output_file_name, "w") as file:
        file.write(f"Classes balanced using SMOTE.\n")
        file.write(f"Class distribution:\n{balanced_df[target_column].value_counts()}\n")
    
    print(f"Classes balanced using SMOTE.\n")
    print(f"Class distribution:\n{balanced_df[target_column].value_counts()}")
    
    return balanced_df

def one_hot_encode_column(df, column_name):
    """
    Transforms a nominal column into numeric using one-hot encoding.

    :param df: pandas DataFrame containing the data.
    :param column_name: Name of the column to encode.
    :return: DataFrame with the one-hot encoded column.
    """
    if column_name not in df.columns:
        raise KeyError(f"The column '{column_name}' does not exist in the DataFrame.")

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=[column_name], prefix=column_name)
    print(f"One-hot encoding applied to column '{column_name}'.")

    return df

def cv_split_and_save(df, test_split=0.2):
    """
    Splits the DataFrame into training and testing sets, saves them to CSV files, and returns the splits.

    Parameters:
    - df: The DataFrame to be split into training and testing sets.
    - test_split: The proportion of the data to be used for testing (default is 0.2 or 20%).
    - file: The name of the file to use for saving the CSV files (default is "extended_df.csv").

    Returns:
    - df_test: The test DataFrame after splitting.
    - df_train: The training DataFrame after splitting.

    The function performs the following:
    1. Splits the DataFrame into train and test sets using `train_test_split`.
    2. Saves both the train and test sets as CSV files.
    3. Returns the train and test DataFrames for further use.
    """
    df_train, df_test = train_test_split(df, test_size=test_split, random_state=42, stratify=df[target_var], shuffle=True)
    df_test.to_csv(os.path.join(ROOT, "validation_" + DS_NAME), index=False)
    df_train.to_csv(os.path.join(ROOT, "train_" + DS_NAME), index=False)
    return df_test, df_train

def standardize(df, vars):
    """
    Applies standard scaling to specified columns (vars) in a DataFrame.
    This function transforms quantitative variables by handling NaN values,
    then saves the transformers used for later use (e.g., for testing the model on new data).

    Parameters:
    - df: DataFrame containing the data to be transformed.
    - vars: List of column names in `df` to which the standard scaling should be applied.

    The function does the following:
    1. Applies standard scaling to each specified column in `vars`.
    2. Skips rows with NaN values while fitting and transforming the data.
    3. Stores the transformers for future use in a pickle file.
    """
    scalers = {}
    for var in vars:
        # Instantiate StandardScaler
        scaler = StandardScaler()
        # Fit the scaler on non-NaN values of the column
        scaler.fit(df[var].dropna().values.reshape(-1, 1))
        mask = ~df[var].isna()
        values = df.loc[mask, var]
        # Transform the values using the fitted scaler
        df.loc[mask, var] = scaler.transform(df.loc[mask, var].values.reshape(-1, 1))
        scalers[var] = scaler

    # Save the scalers to a pickle file for later use
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    return scalers

def powertransform(df, vars):
    """
    Applies Yeo-Johnson power transformation to specified columns (vars) in a DataFrame.
    This function transforms quantitative variables by handling NaN values,
    then saves the transformers used for later use (e.g., for testing the model on new data).

    Parameters:
    - df: DataFrame containing the data to be transformed.
    - vars: List of column names in `df` to which the power transformation should be applied.

    The function does the following:
    1. Applies the Yeo-Johnson power transformation to each specified column in `vars`.
    2. Skips rows with NaN values while fitting and transforming the data.
    3. Stores the transformers for future use in a pickle file.
    """
    q_transformers = {}
    for var in vars:
        # Instantiate PowerTransformer with Yeo-Johnson
        # print(df[var].describe())

        transformer = Pipeline([('standardize', StandardScaler()), ('yeo', PowerTransformer())])
        values_without_nan = df[var].dropna().values.reshape(-1, 1)
        transformer.fit(values_without_nan)
        mask = ~df[var].isna()
        values = df.loc[mask, var]
        values_trans = transformer.transform(values.values.reshape(-1, 1))
        df.loc[mask, var] = values_trans

        # print(df[var].describe())

        q_transformers[var] = transformer

    # Save the transformers to a pickle file. To use maybe in another file if we have to test the test dataframe
    with open(os.path.join(ROOT, "quantitative_transformers.pkl"), "wb") as f:
        pickle.dump(q_transformers, f)

    return q_transformers


def impute_binary_columns(df, verbose=True):
    """
    Impute missing values in binary columns using regression (Logistic Regression).
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing binary columns with missing values.
        verbose (bool): Whether to print a summary of missing data.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    # Identify binary columns
    binary_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
    missing_data = df[binary_columns].isnull().sum()

    # Display summary of missing data if verbose is True
    if verbose:
        print("Missing data summary (binary columns):")
        print(missing_data)

    # Skip if no binary columns have missing values
    if missing_data.sum() == 0:
        if verbose:
            print("No missing values in binary columns.")
        return df

    # Display summary of missing data
    print("Missing data summary:")
    print(missing_data)

    # Configure IterativeImputer with logistic regression
    imputer = IterativeImputer(estimator=RandomForestClassifier(), 
                               max_iter=10, 
                               random_state=42)

    # Apply imputation only to binary columns
    df[binary_columns] = imputer.fit_transform(df[binary_columns])

    # Round values to maintain binary nature
    df[binary_columns] = df[binary_columns].round().astype(int)

    with open('binary_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    return df

def impute_categorical_columns(df, categorical_columns, verbose=True):
    """
    Impute missing values in categorical columns using regression.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns with missing values.
        categorical_columns (list): List of categorical column names to impute.
        verbose (bool): Whether to print a summary of missing data.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    # Check for missing data in specified categorical columns
    missing_data = df[categorical_columns].isnull().sum()

    # Display summary of missing data if verbose is True
    if verbose:
        print("Missing data summary (categorical columns):")
        print(missing_data)

    # Skip if no categorical columns have missing values
    if missing_data.sum() == 0:
        if verbose:
            print("No missing values in categorical columns.")
        return df

    # Encode categorical columns to numeric using OrdinalEncoder
    encoder = OrdinalEncoder()
    df_encoded = df.copy()
    df_encoded[categorical_columns] = encoder.fit_transform(df[categorical_columns].astype(str))

    # Configure IterativeImputer with RandomForestClassifier for categorical data
    imputer = IterativeImputer(estimator=RandomForestClassifier(), 
                               max_iter=10, 
                               random_state=42)
    
    # Fit and transform the data
    df_imputed = imputer.fit_transform(df_encoded)

    # Convert back to categorical values
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    df_imputed[categorical_columns] = encoder.inverse_transform(df_imputed[categorical_columns].round().astype(int))

    # Save the encoder and imputer
    with open('categorical_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    with open('categorical_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    return df_imputed


def remove_outliers(df, quantitative_transforms, quantitative_vars):
    """
    Removes outliers from the given dataframe using domain-specific thresholds
    for oxygen saturation, fever temperature, and age. Saves boxplots and histograms.

    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - quantitative_transforms (dict): Dictionary of transformation objects for quantitative variables.
    - quantitative_vars (list): List of quantitative variables for visualization.
    - ROOT (str): Path to the root directory for saving plots.
    - save_plots_func (callable): Function to save boxplots and histograms.
    """
    # Oxygen saturation outlier removal
    hypoxemia_threshold = 88
    transformed_hypoxemia_threshold = quantitative_transforms["oxygen_saturation"].transform(
        np.array(hypoxemia_threshold).reshape(-1, 1)
    )[0][0]
    df.loc[df["oxygen_saturation"] < transformed_hypoxemia_threshold, "oxygen_saturation"] = np.nan

    # Fever temperature outlier removal
    hypothermia_threshold = 35  # in Celsius
    transformed_hypothermia_threshold = quantitative_transforms["fever_temperature"].transform(
        np.array(hypothermia_threshold).reshape(-1, 1)
    )[0][0]
    df.loc[df["fever_temperature"] <= transformed_hypothermia_threshold, "fever_temperature"] = np.nan

    # Age outlier removal
    min_age, max_age = -1, 95
    transformed_max_age = quantitative_transforms["age"].transform(np.array(max_age).reshape(-1, 1))[0][0]
    if min_age >= -1:
        transformed_min_age = quantitative_transforms["age"].transform(np.array(min_age).reshape(-1, 1))[0][0]
        df.loc[(df["age"] <= transformed_min_age) | (df["age"] >= transformed_max_age), "age"] = np.nan
    else:
        df.loc[df["age"] >= transformed_max_age, "age"] = np.nan

    # Save boxplots and histograms
    save_boxplots_and_histograms(df, quantitative_vars, ROOT, "extended_df_train_boxplot.png")
    return df

if __name__ == "__main__":
    """
    Nominal Variables:
    country_of_residence, sex, history_of_fever, cough, sore_throat, runny_nose, wheezing, shortness_of_breath, headache, loss_of_taste, fatigue_malaise, muscle_aches, joint_pain, diarrhoea, vomiting_nausea, chronic_cardiac_disease, hypertension, chronic_pulmonary_disease, asthma, smoking
    Ordinal Variables:
    date_of_first_symptoms_month, date_of_first_symptoms_dayofyear, admission_date_month, admission_date_dayofyear
    Quantitative Variables:
    age, fever_temperature, oxygen_saturation, date_of_first_symptoms_sin, date_of_first_symptoms_cos, admission_date_sin, admission_date_cos
    """
    df = pd.read_csv(os.path.join(ROOT, DS_NAME))

    nominal_vars = [
        "country_of_residence", "sex", "history_of_fever", "cough", "sore_throat",
        "runny_nose", "wheezing", "shortness_of_breath", "headache", "loss_of_taste",
        "fatigue_malaise", "muscle_aches", "joint_pain", "diarrhoea", "vomiting_nausea",
        "chronic_cardiac_disease", "hypertension", "chronic_pulmonary_disease", "asthma", "smoking"
    ]

    ordinal_vars = [
        "date_of_first_symptoms_month", "date_of_first_symptoms_dayofyear",
        "admission_date_month", "admission_date_dayofyear"
    ]

    quantitative_vars = [
        "age", "fever_temperature", "oxygen_saturation",
        "date_of_first_symptoms_sin", "date_of_first_symptoms_cos",
        "admission_date_sin", "admission_date_cos"
    ]

    target_var = "PCR_result"

    # Separate into train and test. Here, we remove rows without the target variable because they are useless and crash it
    df = df.dropna(subset=[target_var])
    _, df_train = cv_split_and_save(df, test_split=0.2)

    # Apply transformation to quantitative_vars to achieve normal-ish distributions
    quantitative_transforms = powertransform(df_train, quantitative_vars)
    # Plot to check that it is a'ight
    multibar_plots(df_train, target_var, quantitative_vars, [], [], save_name="quantitative_vars_yeo_johnson.png")  # Commented out because it is done

    # Remove outliers
    remove_outliers(df_train, quantitative_transforms, quantitative_vars)

    # Handle missing values
    # Impute binary columns
    df_train = impute_binary_columns(df_train)

    # Impute categorical columns
    # categorical_columns = ['country_of_residence']
    # df_train = impute_categorical_columns(df_train, categorical_columns)
    df_train = df_train.dropna(subset=['country_of_residence'])

    # Std Scaler
    std_scalers = standardize(df_train, quantitative_vars)

    # Apply class balancing
    df_train = balance_classes(df_train, target_var)

    # Apply One-hot encoding to country_of_residence column
    df_train = one_hot_encode_column(df_train, "country_of_residence")


    df_train.to_csv(os.path.join(ROOT, "extended_df_train_preprocessed.csv"), index=False)


