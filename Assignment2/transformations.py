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
from sklearn.impute import SimpleImputer
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
    with open("COVID19_data/scalers.pkl", "wb") as f:
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
    Impute missing values in binary columns with NaNs using Iterative Imputer (Logistic Regression).
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing binary columns with missing values.
        verbose (bool): Whether to print a summary of missing data.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    # Identify binary columns
    binary_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
    
    # Filter only binary columns with NaNs
    binary_columns_with_nans = [col for col in binary_columns if df[col].isnull().any()]
    
    if verbose:
        print("Binary columns identified:", binary_columns)
        print("Binary columns with missing values:", binary_columns_with_nans)
    
    # Skip if no binary columns have missing values
    if not binary_columns_with_nans:
        if verbose:
            print("No binary columns with missing values found.")
        return df

    # Display missing data summary
    missing_data = df[binary_columns_with_nans].isnull().sum()
    if verbose:
        print("Missing data summary (binary columns with NaNs):")
        print(missing_data)

    # Configure IterativeImputer with a RandomForestClassifier
    imputer = IterativeImputer(estimator=RandomForestClassifier(), 
                               max_iter=10, 
                               random_state=42)

    # Apply imputation only to selected binary columns
    df[binary_columns_with_nans] = imputer.fit_transform(df[binary_columns_with_nans])

    # Round values to maintain binary nature
    df[binary_columns_with_nans] = df[binary_columns_with_nans].round().astype(int)

    # Save the trained imputer for future use
    with open('COVID19_data/binary_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    if verbose:
        print("Imputation completed. Binary columns updated.")
    
    return df


def impute_numeric_columns_with_regression(df, numerical_columns, constraints=None):
    """
    Impute missing values in numerical columns using regression, ensuring values are within defined constraints.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing numerical columns with missing values.
        numerical_columns (list): List of numerical column names to impute.
        constraints (dict): Optional. A dictionary defining min and max values for each column.
                            Example: {'oxygen_saturation': (50, 100), 'fever_temperature': (35, 42)}
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    if constraints is None:
        constraints = {}

    for col in numerical_columns:
        print(f"\nProcessing column: {col}")
        
        # Create mask for missing values
        missing_mask = df[col].isnull()
        if missing_mask.sum() > 0:
            non_null = df[~missing_mask]  # Rows where col is not null
            null = df[missing_mask]      # Rows where col is null
            
            # Ensure there are enough rows to train the regression model
            if len(non_null) > 10 and len(non_null.columns) > 1:
                X_train = non_null.drop(columns=[col])
                y_train = non_null[col]
                
                # Impute missing values in X_train (predictors)
                imputer = SimpleImputer(strategy="mean")
                X_train = imputer.fit_transform(X_train)
                
                # Train regression model
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                
                # Prepare predictors for rows with missing values
                X_predict = null.drop(columns=[col])
                X_predict = imputer.transform(X_predict)
                
                # Predict and constrain values
                predicted = reg.predict(X_predict)
                
                # Apply constraints if defined
                if col in constraints:
                    min_val, max_val = constraints[col]
                    predicted = np.clip(predicted, min_val, max_val)
                
                # Fill missing values
                df.loc[missing_mask, col] = predicted
                print(f"Imputed missing values in {col} using regression.")
            else:
                print(f"Not enough data to impute {col} using regression. Skipping...")
        else:
            print(f"No missing values in {col}.")
    
    return df


def impute_dates_by_order(df, date_prefix):
    """
    Impute missing values in date-related columns based on chronological order 
    and replace the original datetime column with numeric (timestamp) values.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing date-related columns.
        date_prefix (str): Prefix of the date columns (e.g., 'date_of_first_symptoms' or 'admission_date').
    
    Returns:
        pd.DataFrame: DataFrame with missing date columns imputed and recalculated.
    """
    # Define columns
    year_col = f"{date_prefix}_year"
    month_col = f"{date_prefix}_month"
    day_col = f"{date_prefix}_dayofmonth"
    dayofyear_col = f"{date_prefix}_dayofyear"
    sin_col = f"{date_prefix}_sin"
    cos_col = f"{date_prefix}_cos"
    date_col = f"{date_prefix}_date"

    # Interpolate missing values in base columns (year, month, day)
    for col in [year_col, month_col, day_col]:
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method='linear')

    # Reconstruct full dates
    df[date_col] = pd.to_datetime(
        dict(year=df[year_col], month=df[month_col], day=df[day_col]),
        errors='coerce'
    )

    # Interpolate the full date if necessary
    if df[date_col].isnull().any():
        df[date_col] = df[date_col].interpolate(method='time')

    # Replace datetime column with timestamp
    df[date_col] = df[date_col].apply(
        lambda x: x.timestamp() if pd.notnull(x) else np.nan
    )

    # Recalculate derived columns
    df[dayofyear_col] = pd.to_datetime(df[date_col], unit='s').dt.dayofyear
    df[sin_col] = np.sin(2 * np.pi * df[dayofyear_col] / 365.25)
    df[cos_col] = np.cos(2 * np.pi * df[dayofyear_col] / 365.25)

    return df


def remove_outliers(df, quantitative_vars):
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
    df.loc[df["oxygen_saturation"] < hypoxemia_threshold, "oxygen_saturation"] = np.nan

    # Fever temperature outlier removal
    hypothermia_threshold = 35  # in Celsius
    df.loc[df["fever_temperature"] <= hypothermia_threshold, "fever_temperature"] = np.nan

    # Age outlier removal
    min_age, max_age = -1, 95
    if min_age >= -1:
        df.loc[(df["age"] <= min_age) | (df["age"] >= max_age), "age"] = np.nan
    else:
        df.loc[df["age"] >= max_age, "age"] = np.nan

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
    #quantitative_transforms = powertransform(df_train, quantitative_vars)
    # Plot to check that it is a'ight
    multibar_plots(df_train, target_var, quantitative_vars, [], [], save_name="quantitative_vars_yeo_johnson.png")  # Commented out because it is done

    # Remove outliers
    remove_outliers(df_train, quantitative_vars)

    # Handle missing values
    # Impute binary columns
    df_train = impute_binary_columns(df_train)

    missing_values = df_train.isnull().sum()

    # Show columns with missing values
    remaining_missing = missing_values[missing_values > 0]
    print(remaining_missing)

    #categorical_columns = ['country_of_residence', 'patient_id']
    numerical_columns = ['oxygen_saturation', 'fever_temperature']
    date_columns = [col for col in df_train.columns if 'date_of_first_symptoms' in col or 'admission_date' in col]

    # FOR NOW, JUST DROP MISSING VALUES (2) FOR PATIENT_ID
    df_train = df_train.dropna(subset=['patient_id'])

    # Fill with mode (2)
    df_train['country_of_residence'] = df_train['country_of_residence'].fillna(df_train['country_of_residence'].mode()[0])

    # Fill age (18) with mean or mode
    age_mode = df_train['age'].mode()[0]
    print(age_mode)
    df_train['age'] = df_train['age'].fillna(age_mode)

    # Definir límites razonables para las variables
    constraints = {
        'oxygen_saturation': (50, 100),
        'fever_temperature': (35, 42)
    }
    
    df_train = impute_numeric_columns_with_regression(df_train, numerical_columns, constraints=constraints)

    # Impute date columns
    df_train = impute_dates_by_order(df_train, "date_of_first_symptoms")

    df_train = impute_dates_by_order(df_train, "admission_date")

    # Check for missing values
    missing_values = df_train.isnull().sum()
    remaining_missing = missing_values[missing_values > 0]

    if not remaining_missing.empty:
        print("Columns with missing values after imputation:")
        print(remaining_missing)
    else:
        print("No columns with missing values after imputation.")


    # Apply class balancing
    df_train = balance_classes(df_train, target_var)

    # Apply One-hot encoding to country_of_residence column
    df_train = one_hot_encode_column(df_train, "country_of_residence")

    # Std Scaler
    # std_scalers = standardize(df_train, quantitative_vars+ordinal_vars)

    df_train.drop('Unnamed: 0', axis=1, inplace=True)
    df_train.to_csv(os.path.join(ROOT, "extended_df_train_preprocessed.csv"), index=False)
    # df_train.to_csv(os.path.join(ROOT, "extended_df_train_preprocessed_standard.csv"), index=False)