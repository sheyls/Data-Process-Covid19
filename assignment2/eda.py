import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


root = "./COVID19_data/"

file1 = "hospital1.xlsx"
file2 = "hospital2.xlsx"

df1 = pd.read_excel(root + file1)
df2 = pd.read_excel(root + file2)

target_var = "PCR_result"
predictor_vars1 = list(df1.columns)
predictor_vars2 = list(df2.columns)
predictor_vars1.remove(target_var)
predictor_vars2.remove(target_var)

quantitative_vars1 = ["age", "fever_temperature", "oxygen_saturation"]
ordinal_vars1 = []
time_vars1 = ["date_of_first_symptoms", "BASVURUTARIHI"]
identifiers1 = ["admission_id", "patient_id"]
non_nominal_vars = set(quantitative_vars1 + ordinal_vars1 + time_vars1 + identifiers1)
nominal_vars1 = [col for col in df1.columns if col not in non_nominal_vars]

quantitative_vars2 = ["age", "fever_temperature", "oxygen_saturation"]
ordinal_vars2 = []
time_vars2 = ["date_of_first_symptoms", "admission_date"]
identifiers2 = ["patient ID", "patient ID.1"]
non_nominal_vars = set(quantitative_vars2 + ordinal_vars2 + time_vars2 + identifiers2)
nominal_vars2 = [col for col in df2.columns if col not in non_nominal_vars]


def print_description_and_info(df1, df2, file='dataframes_summary.txt', root=root):
    # Assuming df1 and df2 are pandas DataFrames
    with open(root + file, 'w') as file:
        # Write df1.describe() to the file
        file.write("DF1 - Describe:\n")
        file.write(df1.describe().to_string())
        file.write("\n\n")

        # Write df1.info() to the file
        file.write("DF1 - Info:\n")
        df1.info(buf=file)
        file.write("\n\n")

        # Write df2.describe() to the file
        file.write("DF2 - Describe:\n")
        file.write(df2.describe().to_string())
        file.write("\n\n")

        # Write df2.info() to the file
        file.write("DF2 - Info:\n")
        df2.info(buf=file)
        file.write("\n")

        df1_not_in_df2, df2_not_in_df1 = compare_columns(df1, df2)
        file.write("Columns in df1 but not in df2:\n")
        for col in df1_not_in_df2:
            file.write(f"{col}: {df1[col].dtype}\n")
        file.write("\n")

        # Write columns in df2 but not in df1
        file.write("Columns in df2 but not in df1:\n")
        for col in df2_not_in_df1:
            file.write(f"{col}: {df2[col].dtype}\n")
        file.write("\n")
"""
quantitative_vars1 = ["age", "fever_temperature", "oxygen_saturation"]
time_vars1 = ["date_of_first_symptoms", "admission_date"]
identifiers = ["admission_id", "patient_id"]
nominal_vars1 = others
"""
# Column Plot: Comparing the target variable in both DataFrames
def multibar_plots(dataframe, response_variable, quantitative_vars, time_vars, nominal_vars, root=root, save_name=""):
    """
    Creates multi-bar plots comparing the distribution of each predictor variable across the classes
    of the response variable.

    Parameters:
    - dataframe: pd.DataFrame
        The input dataframe containing the response and predictor variables.
    - response_variable: str
        The name of the response variable (target) column.
    - predictors: list of str
        List of predictor variable column names to plot.

    Returns:
    - None
    """
    unique_classes = dataframe[response_variable].unique()
    n_classes = len(unique_classes)

    # Set up the plot grid
    predictors = quantitative_vars + time_vars + nominal_vars
    n_predictors = len(predictors)
    n_cols = 3
    n_rows = -(-n_predictors // n_cols)  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, predictor in enumerate(predictors):
        ax = axes[i]
        if predictor in quantitative_vars:
            # For numerical predictors, create bins and plot counts per class
            dataframe['binned'] = pd.cut(dataframe[predictor], bins=10)
            sns.countplot(data=dataframe, x='binned', hue=response_variable, ax=ax)
            ax.set_title(f'Distribution of {predictor} by {response_variable}')
            dataframe.drop(columns='binned', inplace=True)
        elif predictor in nominal_vars:
            # For nominal predictors, plot counts per class
            sns.countplot(data=dataframe, x=predictor, hue=response_variable, ax=ax)
            ax.set_title(f'Distribution of {predictor} by {response_variable}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yscale('log')
        else: # time_vars
            dataframe[predictor + '_year'] = dataframe[predictor].dt.year
            dataframe[predictor + '_month'] = dataframe[predictor].dt.month
            dataframe[predictor + '_day'] = dataframe[predictor].dt.day

            # Choose the component you want to plot by (you can modify this line)
            # For example, let's plot by 'month'
            sns.countplot(data=dataframe, x=predictor + '_month', hue=response_variable, ax=ax)

            ax.set_title(f'Distribution of {predictor} by {response_variable}')

            # Set y-axis to logarithmic scale
            ax.set_yscale('log')

            # Optionally, set the x-axis labels to more meaningful ones if you're using months
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.tick_params(axis='x', labelrotation=45)
    ax.legend(title=response_variable)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(root + save_name)
    # plt.show()


def compare_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    # Find columns in df1 but not in df2
    df1_not_in_df2 = df1.columns.difference(df2.columns).tolist()

    # Find columns in df2 but not in df1
    df2_not_in_df1 = df2.columns.difference(df1.columns).tolist()

    return df1_not_in_df2, df2_not_in_df1

# VISUALIZATION EDA
print_description_and_info(df1, df2) # Commented out because it is done
multibar_plots(df1, target_var, quantitative_vars1, time_vars1, nominal_vars1, save_name="df1_plot.png")
multibar_plots(df2, target_var, quantitative_vars2, time_vars2, nominal_vars2, save_name="df2_plot.png")

# FIXME PLOT CORRELATION OF THE QUANTITATIVE VARIABLES
# FIXME PLOT CHI-SQUARE TEST FOR NOMINAL VARIABLES


# PREPROCESSING
# FIXME CHANGE THE NAMES OF THE COLUMNS THAT ARE OBVIOUSLY THE SAME ONE IN BOTH DATAFRAMES
# FIXME CHANGE THE TYPES SO IT MATCHES THE TYPE OF VARIABLE (categorical to int64 with embedding for strings)
# FIXME MERGE DATAFRAMES
#      CONSIDER SAME PATIENTS WITH DIFFERENT DATES
#      CHECK THAT COLUMNS OVERLAP
# FIXME CHANGE THE DATES TO A USABLE FORMAT
#      OPTION 1: Treat a year as a cycle and embed it with two polar coordinates
#      OPTION 2: Separate it into day of the hour, day of the month, day of the week, month, year, etc.
#                This would require checking importance again with chi-square to decide which to keep
# FIXME SCALE QUANTITATIVE VARIABLES


