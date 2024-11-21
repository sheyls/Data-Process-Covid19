"""
This file should contain the scalings and other transformations that are parametric and require using only the training set.


## EDA
- [x] Analizar el balance del target


## Preproc
- [X] Separar en train-test
- [X] transformaciones para ponernos mas normales (log)
- [Fran] outliers (detect and remove)
- [ ] missings (fill nan's)
- [x] transformar pais a numerico (vector)
- [SHEYLA ] Balancear las clases < 15 - 85 por ahora

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
from caffe2.python.examples.imagenet_trainer import Train
from docutils.frontend import validate_encoding_and_error_handler
from pandas.core.common import random_state
from scipy.special import ellip_harm
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from sklearn.preprocessing import PowerTransformer
from spacy.cli import train

from Assignment2.eda import multibar_plots

ROOT = "./COVID19_data/"
DS_NAME = "extended_df.csv"

import pandas as pd

def balance_classes_with_smote(df):
    X = df.drop(columns)
    balancer = SMOTE(random_state=10)
    X_balanced

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
    df_test.to_csv(os.path.join(ROOT, "test_" + DS_NAME), index=False)
    df_train.to_csv(os.path.join(ROOT, "train_" + DS_NAME), index=False)
    return df_test, df_train

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
        yeo_johnson_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yeo_johnson_transformer.fit(df[var].dropna().values.reshape(-1, 1))
        mask = ~df[var].isna()
        values = df.loc[mask, var]
        df.loc[mask, var] = yeo_johnson_transformer.transform(df.loc[mask, var].values.reshape(-1, 1))
        q_transformers[var] = yeo_johnson_transformer

    # Save the transformers to a pickle file. To use maybe in another file if we have to test the test dataframe
    with open("quantitative_transformers.pkl", "wb") as f:
        pickle.dump(q_transformers, f)

    return q_transformers


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
    multibar_plots(df, target_var, quantitative_vars, [], [], save_name="quantitative_vars_yeo_johnson.png")  # Commented out because it is done


    # OUTLIER REMOVAL (I have to transform the value cuttoff point too or move it before transformations)
    # Oxygen detection filter by absurd levels using domain knowledge
    # Below 92 it is hypoxemia and requires help right away. I am looking at what is absurd in someone concious
    # Down to a limit of 88 for groups of people with lung problems (COPD)
    # We remove those below that because they would have lost conciousness and would have been put on oxygen to rise it up.
    df.loc[df["oxygen_saturation"] < quantitative_transforms["oxygen_saturation"].transform(np.array(88).reshape(-1, 1)), "oxygen_saturation"] = np.nan

    # Temperature
    # Hypothermia: A body temperature below 95°F (35°C)
    df.loc[df["fever_temperature"] < quantitative_transforms["fever_temperature"].transform(np.array(35).reshape(-1, 1)), "fever_temperature"] = np.nan

    # Age
    # I do not feel confident just pruning babies
    min_age, max_age = (-1, 95)
    min_age, max_age = quantitative_transforms["age"].transform(np.array(min_age).reshape(-1, 1)), quantitative_transforms["age"].transform(np.array(max_age).reshape(-1, 1))
    df.loc[min_age >= df["age"] | df["age"] >= max_age, "age"] = np.nan
    

    # Apply One-hot encoding to country_of_residence column
    df_train = one_hot_encode_column(df_train, "country_of_residence")
