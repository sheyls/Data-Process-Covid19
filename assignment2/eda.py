import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp, ranksums

from sklearn.preprocessing import RobustScaler

ROOT = "./COVID19_data/"

FILE_1 = "hospital1.xlsx"
FILE_2 = "hospital2.xlsx"


def print_description_and_info(df1, df2=None, file='dataframes_summary.txt', root=ROOT):
    # Assuming df1 and df2 are pandas DataFrames
    with open(os.path.join(root, file), 'w') as file:
        # Write df1.describe() to the file
        file.write("DF1 - Describe:\n")
        file.write(df1.describe().to_string())
        file.write("\n\n")

        # Write df1.info() to the file
        file.write("DF1 - Info:\n")
        df1.info(buf=file)
        file.write("\n\n")

        if df2 is not None:
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
def multibar_plots(dataframe, response_variable, quantitative_vars, time_vars, nominal_vars, root=ROOT, save_name=""):
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
            # ax.set_xticks(range(1, 13))
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
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
    plt.savefig(os.path.join(root, save_name))
    # plt.show()


def compare_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    # Find columns in df1 but not in df2
    df1_not_in_df2 = df1.columns.difference(df2.columns).tolist()

    # Find columns in df2 but not in df1
    df2_not_in_df1 = df2.columns.difference(df1.columns).tolist()

    return df1_not_in_df2, df2_not_in_df1

def chi_square_test(df, nominal_vars, target_var, alpha=0.05, output_file="chi_square_results.txt", root=ROOT):
    results = []

    for var in nominal_vars:
        # Create a contingency table
        aux_df = df[[var, target_var]].dropna()
        contingency_table = pd.crosstab(aux_df[var], aux_df[target_var])

        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Determine significance
        significant = p < alpha

        # Append results
        results.append((var, chi2, p, significant))

    results_sorted_by_pvalue = sorted(results, key=lambda x: x[2])
    results_sorted_by_var = sorted(results, key=lambda x: x[0])

    # Write results to a file
    with open(os.path.join(root, output_file), "w") as file:
        # file.write(f"Chi-Square Test Results (alpha={alpha}):\n")
        # file.write("Variable\tChi2\tp-value\tSignificant\n")
        # for var, chi2, p, significant in results:
        #     file.write(f"{var}\t{chi2:.4f}\t{p:.4e}\t{significant}\n")

        file.write(f"Chi-Square Test Results (alpha={alpha}):\n\n")
        file.write(f"{'Variable':<30} {'Chi2':<15} {'p-value':<15} {'Significant':<10}\n")
        file.write(f"{'-' * 70}\n")
        for var, chi2, p, significant in results_sorted_by_pvalue:
            file.write(f"{var:<30} {chi2:<15.4f} {p:<15.4e} {str(significant):<10}\n")
        file.write(f"{'-' * 70}\n")
        for var, chi2, p, significant in results_sorted_by_var:
            file.write(f"{var:<30} {chi2:<15.4f} {p:<15.4e} {str(significant):<10}\n")

    return results

def calculate_correlations(df1, quantitative_vars1, df2=None, quantitative_vars2=None, file1="correlations_df2.csv", file2="correlations_df1.csv"):
    # Ensure only quantitative variables are used
    df1_quant = df1[quantitative_vars1]
    corr_df1 = df1_quant.corr()
    corr_df1.to_csv(os.path.join(ROOT, file1))

    if df2 is not None:
        df2_quant = df2[quantitative_vars2]
        corr_df2 = df2_quant.corr()
        corr_df2.to_csv(os.path.join(ROOT, file2))

        return corr_df1, corr_df2
    return corr_df1


def save_boxplots_and_histograms(df, quantitative_vars, save_dir, file_name, percentile=(0.25,99.75)):
    """
    Create boxplots and histograms side by side for each column in quantitative_vars and save the figure.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - quantitative_vars (list): A list of column names to create plots for.
    - save_dir (str): The directory to save the figure.
    - file_name (str): The name of the output file (e.g., 'boxplots_histograms.png').
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    # Number of variables
    num_vars = len(quantitative_vars)

    # Create a figure with 2 subplots for each variable (boxplot + histogram)
    fig, axes = plt.subplots(num_vars, 3, figsize=(18, 6 * num_vars), constrained_layout=True)

    # Ensure axes are iterable for a single column
    if num_vars == 1:
        axes = [axes]

    # Generate boxplot and histogram for each variable
    for idx, predictor in enumerate(quantitative_vars):
        boxplot_ax = axes[idx][0] if num_vars > 1 else axes[0]
        hist_ax = axes[idx][1] if num_vars > 1 else axes[1]
        outlier_hist_ax = axes[idx][2] if num_vars > 1 else axes[2]

        # Boxplot
        sns.boxplot(y=df[predictor], color="skyblue", ax=boxplot_ax)
        boxplot_ax.set_title(f'Boxplot of {predictor}')
        boxplot_ax.set_ylabel(predictor)
        boxplot_ax.set_xlabel('Values')

        # Histogram
        sns.histplot(df[predictor], kde=True, color="skyblue", ax=hist_ax)
        hist_ax.set_title(f'Histogram of {predictor}')
        hist_ax.set_xlabel('Values')
        hist_ax.set_ylabel('Frequency')

        data = df[predictor].dropna().values
        a, b = np.percentile(data, percentile)
        bins = np.linspace(data.min(), data.max(), 50)
        outlier_hist_ax.hist(data[(data >= a) & (data <= b)], bins=bins, color='blue', label='1-99 Percentile')
        outlier_hist_ax.hist(data[data < a], bins=bins, color='red', label='Below  0.25 Percentile')
        outlier_hist_ax.hist(data[data > b], bins=bins, color='red', label='Above 99.75 Percentile')
        outlier_hist_ax.set_title(f'Outlier Highlighted Histogram of {predictor}')
        outlier_hist_ax.set_xlabel('Values')
        outlier_hist_ax.set_ylabel('Frequency')
        outlier_hist_ax.legend()

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Boxplots and histograms saved to: {save_path}")


def classify_and_write(df, nominal_columns, ordinal_columns, quantitative_columns, file_name):
    # Prepare the text to write to file
    output = f"Nominal Variables:\n{', '.join(nominal_columns) if nominal_columns else 'None'}\n\n"
    output += f"Ordinal Variables:\n{', '.join(ordinal_columns) if ordinal_columns else 'None'}\n\n"
    output += f"Quantitative Variables:\n{', '.join(quantitative_columns) if quantitative_columns else 'None'}\n"

    # Write to a text file
    with open(file_name, 'w') as f:
        f.write(output)


def print_non_param_homogeneity_tests(df, quantitative_vars, target_column, file_path):
    with open(file_path, 'w') as f:
        for var in quantitative_vars:
            # Split the data into two groups based on the target variable
            class_0 = df[df[target_column] == 0][var].dropna()
            class_1 = df[df[target_column] == 1][var].dropna()

            # Perform Kolmogorov-Smirnov Test
            ks_stat, ks_p_value = ks_2samp(class_0, class_1)

            # Perform Wilcoxon Rank-Sum Test
            w_stat, w_p_value = ranksums(class_0, class_1)

            # Pretty print the results
            f.write(f"Significance Tests for '{var}':\n")
            f.write(f"  Kolmogorov-Smirnov Test p-value: {ks_p_value:.3e}\n")
            f.write(f"  Wilcoxon Rank-Sum Test p-value: {w_p_value:.3e}\n")
            f.write("-" * 50 + "\n")


def print_class_balance(df, target_var, file):
    """
    Prints the class balance of the target variable in the given dataframe and optionally writes it to a file.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    target_var (str): The name of the target variable column.
    file (str, optional): Path to the file where the class balance should be saved. Defaults to None.

    Returns:
    None
    """
    class_counts = df[target_var].value_counts()
    class_percentages = df[target_var].value_counts(normalize=True) * 100

    # Prepare the class balance output
    balance_output = "\n".join(
        f"Class '{cls}': {count} samples ({percentage:.2f}%)"
        for cls, count, percentage in zip(class_counts.index, class_counts, class_percentages)
    )

    # Print the results
    print("Class Balance:")
    print(balance_output)

    # Write to file if specified
    if file:
        with open(file, 'w') as f:
            f.write("Class Balance:\n")
            f.write(balance_output)


if __name__ == '__main__':
    df1 = pd.read_excel(os.path.join(ROOT, FILE_1))
    df2 = pd.read_excel(os.path.join(ROOT, FILE_2))

    target_var = "PCR_result"
    predictor_vars1 = list(df1.columns)
    predictor_vars2 = list(df2.columns)
    predictor_vars1.remove(target_var)
    predictor_vars2.remove(target_var)

    identifiers1 = ["patient ID", "patient ID.1"]
    identifiers2 = ["admission_id", "patient_id"]

    quantitative_vars1 = ["age", "fever_temperature", "oxygen_saturation"]
    ordinal_vars1 = []
    time_vars1 = ["date_of_first_symptoms", "BASVURUTARIHI"]
    non_nominal_vars = set(quantitative_vars1 + ordinal_vars1 + time_vars1 + identifiers1 + [target_var])
    nominal_vars1 = [col for col in df1.columns if col not in non_nominal_vars]

    quantitative_vars2 = ["age", "fever_temperature", "oxygen_saturation"]
    ordinal_vars2 = []
    time_vars2 = ["date_of_first_symptoms", "admission_date"]
    non_nominal_vars = set(quantitative_vars2 + ordinal_vars2 + time_vars2 + identifiers2 + [target_var])
    nominal_vars2 = [col for col in df2.columns if col not in non_nominal_vars]

    # VISUALIZATION EDA
    # print_description_and_info(df1, df2) # Commented out because it is done
    # multibar_plots(df1, target_var, quantitative_vars1, time_vars1, nominal_vars1, save_name="df1_plot.png") # Commented out because it is done
    # multibar_plots(df2, target_var, quantitative_vars2, time_vars2, nominal_vars2, save_name="df2_plot.png") # Commented out because it is done

    # PLOT CORRELATION OF THE QUANTITATIVE VARIABLES
    # calculate_correlations(df1, quantitative_vars1, df2, quantitative_vars2) # commented out because it is done

    # PLOT CHI-SQUARE TEST FOR NOMINAL VARIABLES
    # chi_square_test(df1, nominal_vars1, target_var, alpha=0.05, output_file="chi_square_results_df1.txt") # Commented out because it is done
    # chi_square_test(df2, nominal_vars2, target_var, alpha=0.05, output_file="chi_square_results_df2.txt") # Commented out because it is done

    # PREPROCESSING
    # FIXME CHANGE THE NAMES OF THE COLUMNS THAT ARE OBVIOUSLY THE SAME ONE IN BOTH DATAFRAMES
    #       DEAL WITH THE BASVURUTARIHI and admission_date
    name_mapping = {
        'gender K=female E=male': 'sex',
        'nationality': 'country_of_residence',
        'patient ID': 'patient_id',
        'BASVURUTARIHI': 'admission_date'
    }
    df1.rename(columns=name_mapping, inplace=True)
    quantitative_vars1 = ["age", "fever_temperature", "oxygen_saturation"]
    ordinal_vars1 = []
    time_vars1 = ["date_of_first_symptoms", "admission_date"]
    identifiers1 = ["patient_id", "patient ID.1"]
    non_nominal_vars = set(quantitative_vars1 + ordinal_vars1 + time_vars1 + identifiers1 + [target_var])
    nominal_vars1 = [col for col in df1.columns if col not in non_nominal_vars]

    # CHANGE THE TYPES SO IT MATCHES THE TYPE OF VARIABLE (categorical to int64 with embedding for strings)
    unique_values1 = df1['country_of_residence'].unique()
    unique_values2 = df2['country_of_residence'].unique()
    merged_unique_values = list(set(list(unique_values1) + list(unique_values2)))
    merged_unique_values = [x for x in merged_unique_values if pd.notna(x)]
    merged_unique_values.sort()
    country_mapping = {}
    for i, country in enumerate(merged_unique_values):
        country_mapping[country] = i
    df1['country_of_residence'] = df1['country_of_residence'].map(country_mapping)
    df2['country_of_residence'] = df2['country_of_residence'].map(country_mapping)

    gender_mapping = {"K": 1, "E": 0}
    df1["sex"] = df1["sex"].map(gender_mapping)
    df2["sex"] = df2["sex"].map(gender_mapping)

    pcr_mapping = {"positive": 1, "negative": 0}
    df1[target_var] = df1[target_var].map(pcr_mapping)
    df2[target_var] = df2[target_var].map(pcr_mapping)

    # CHANGING THE TYPE IS ONLY VIABLE WITHOUT NULLS THERE
    # for col in nominal_vars1:
    #     if df1[col].dtype != 'object':  # Check if the column is not of type object
    #         df1[col] = df1[col].fillna(-2).astype('int64')  # Convert to int64
    # for col in nominal_vars2:
    #     if df2[col].dtype != 'object':  # Check if the column is not of type object
    #         df2[col] = df2[col].fillna(-2).astype('int64')  # Convert to int64
    # df1 = df1.replace(-2, np.nan)
    # df2 = df2.replace(-2, np.nan)

    # print_description_and_info(df1, df2, file="updated_dataframes_summary.txt") # Commented out because it is done

    df1.to_csv(os.path.join(ROOT, "updated_df1.csv"))
    df2.to_csv(os.path.join(ROOT, "updated_df2.csv"))

    # FIXME MERGE DATAFRAMES
    #       DONE COLUMN WISE MERGE
    #           CHECK THAT COLUMNS OVERLAP
    #       FIXME TUPLE MERGE AS REQUIRED
    cols = set(df1.columns)
    cols = list(cols.intersection(df2.columns))
    df1_premerge = df1[cols]
    df2_premerge = df2[cols]
    df = pd.concat([df1_premerge, df2_premerge], ignore_index=True)

    # print_description_and_info(df, file="merged_df_description.txt") # Commented out because it is done
    # multibar_plots(df, target_var, quantitative_vars2, time_vars2, nominal_vars2, save_name="merged_df_plot.png") # Commented out because it is done
    # calculate_correlations(df, quantitative_vars2, file1="merged_correlation.txt") # commented out because it is done
    # chi_square_test(df, nominal_vars2, target_var, alpha=0.05, output_file="merged_chi_square_results_df.txt") # Commented out because it is done
    print_non_param_homogeneity_tests(df, quantitative_vars2, target_var, os.path.join(ROOT, "merged_quantitative_tests.txt"))
    print_class_balance(df, target_var, file=os.path.join(ROOT, "merged_class_balance.txt"))
    # df.to_csv(os.path.join(ROOT, "merged_cols_df.csv"))

    # CHANGE THE DATES TO A USABLE FORMAT
    #      OPTION 1: Treat a year as a cycle and embed it with two polar coordinates
    new_quantitative = []
    for predictor in time_vars2:
        dayofyear = df[predictor].dt.dayofyear
        sin = np.sin(2 * np.pi * dayofyear / 365.0)
        cos = np.cos(2 * np.pi * dayofyear / 365.0)
        df[predictor + "_sin"] = sin
        df[predictor + "_cos"] = cos
        new_quantitative += [predictor + "_sin", predictor + "_cos"]
    quantitative_vars2 += new_quantitative

    #      OPTION 2: Separate it into day of the hour, day of the month, day of the week, month, year, etc.
    #                This would require checking importance again with chi-square to decide which to keep
    new_ordinal = []
    for predictor in time_vars2:
        df[predictor + '_year'] = df[predictor].dt.year
        df[predictor + '_month'] = df[predictor].dt.month
        df[predictor + '_dayofmonth'] = df[predictor].dt.day
        df[predictor + '_dayofyear'] = df[predictor].dt.dayofyear
        df[predictor + '_dayofweek'] = df[predictor].dt.dayofweek
        new_ordinal += [predictor + '_year',predictor + '_month',predictor + '_dayofmonth',predictor + '_dayofyear',predictor + '_dayofweek']


    ordinal_vars2 += new_ordinal

    # SCALE QUANTITATIVE VARIABLES AND BOXPLOTS
    # save_boxplots_and_histograms(df, quantitative_vars=quantitative_vars2, save_dir=ROOT, file_name='merged_boxplots.png')

    # REMOVE OLD TIME VARIABLES
    df = df.drop(columns=time_vars2)

    # SAVE THE OUTPUT AND PRINT LAST DESCRIPTIONS AND PLOTS
    # print_description_and_info(df, file="extended_df_description.txt")
    # multibar_plots(df, target_var, quantitative_vars2, [], nominal_vars2 + ordinal_vars2,
    #                save_name="extended_df_multibar_plot.png")


    # FINAL TESTS
    chi_square_test(df, nominal_vars=nominal_vars2 + ordinal_vars2, target_var=target_var, output_file="extended_chi_squared_results.txt")
    calculate_correlations(df, quantitative_vars2, file1="extended_correlations.txt")
    print_non_param_homogeneity_tests(df, quantitative_vars2, target_var, os.path.join(ROOT, "extended_quantitative_tests.txt"))
    classify_and_write(df, nominal_vars2, ordinal_vars2, quantitative_vars2, os.path.join(ROOT, 'extended_variables_classification.txt'))
    print_class_balance(df, target_var, file=os.path.join(ROOT, "extended_class_balance.txt"))
    df.to_csv(os.path.join(ROOT, "extended_df.csv"), index=False)


    # FINAL VARIABLE SELECTION USING EDA (DISCARDING IRRELEVANT VARIABLES)
    pvalue_treshold = 0.1 # Slightly more relaxed than the original 0.5
    final_nominal_vars = []
    for var in nominal_vars2:
        # Create a contingency table
        aux_df = df[[var, target_var]].dropna()
        contingency_table = pd.crosstab(aux_df[var], aux_df[target_var])

        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Determine significance
        if p < pvalue_treshold:
            final_nominal_vars.append(var)

    final_ordinal_vars = []
    manual_filter = [
        # EXAMPLE OF MANUALLY DISCARDED FEATURES DISREGARDING TESTS
        'admission_date_dayofweek',
        'date_of_first_symptoms_dayofweek',
        'admission_date_year',
        'date_of_first_symptoms_year',
        'admission_date_dayofmonth',
        'date_of_first_symptoms_dayofmonth']

    for var in ordinal_vars2:
        if var in manual_filter:
            continue
        print(var)

        # Create a contingency table
        aux_df = df[[var, target_var]].dropna()
        contingency_table = pd.crosstab(aux_df[var], aux_df[target_var])

        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Determine significance
        if p < pvalue_treshold:
            final_ordinal_vars.append(var)

    # They are not highly correlated and the quantitative variable significance tests show all <0.1
    # Only fever_temperature is >0.05 (~0.097)
    final_quantitative_vars = quantitative_vars2

    classify_and_write(df, final_nominal_vars, final_ordinal_vars, final_quantitative_vars, os.path.join(ROOT, 'preliminary_selected_variables_classification.txt'))











