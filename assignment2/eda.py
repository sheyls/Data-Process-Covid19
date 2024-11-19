import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


root = "./"

file1 = "hospital1.xlsx"
file2 = "hospital2.xlsx"

df1 = pd.read_excel(root + file1)
df2 = pd.read_excel(root + file2)

print(df1.describe())
print(df2.describe())
print(df1.info())
print(df2.info())


target_var = "PCR_result"
predictor_vars1 = list(df1.columns)
predictor_vars2 = list(df2.columns)
predictor_vars1.remove(target_var)
predictor_vars2.remove(target_var)


target_var = "PCR_result"
predictor_vars1 = list(df1.columns)
predictor_vars2 = list(df2.columns)
predictor_vars1.remove(target_var)
predictor_vars2.remove(target_var)


# Column Plot: Comparing the target variable in both DataFrames
def compare_variable(df1, df2, target_var):
    fig, ax = plt.subplots(figsize=(10, 6))
    df1[target_var].value_counts().sort_index().plot(kind='bar', color='blue', alpha=0.5, ax=ax, label='df1')
    df2[target_var].value_counts().sort_index().plot(kind='bar', color='orange', alpha=0.5, ax=ax, label='df2')
    ax.set_title(f"Comparison of {target_var} in df1 and df2")
    ax.set_xlabel(target_var)
    ax.set_ylabel("Count")
    ax.legend()
    plt.show()
    # plt.savefig("./fig/")

# Call the functions
compare_all_columns(df1, df2, predictor_vars1, predictor_vars2)
compare_target_variable(df1, df2, target_var)
