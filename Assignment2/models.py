import pickle

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import wittgenstein as lw


class Models():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X_train, y_train):
        criterion = ['gini', 'entropy']
        max_depth = [5, 10, 15]
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for criteria in criterion:
                model = DecisionTreeClassifier(criterion=criteria, max_depth=depth, random_state=42)
                for scoring in scores_dict.keys():
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring].append(np.mean(scores))
        return scores_dict

    def rule_induction(self, X_train, y_train):
        K = [1, 2, 3, 4]
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for k in K:
            model = lw.RIPPER(k=k)
            for scoring in scores_dict.keys():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring].append(np.mean(scores))
        return scores_dict

    def logistic_regression(self, X_train, y_train):
        penalties = ['none', 'l1', 'l2']
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for penalty in penalties:
            model = LogisticRegression(penalty=penalty, max_iter=1000, random_state=42)
            for scoring in scores_dict.keys():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring].append(np.mean(scores))
        return scores_dict

    def svm(self, X_train, y_train):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for kernel in kernels:
            model = SVC(kernel=kernel)
            for scoring in scores_dict.keys():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring].append(np.mean(scores))
        return scores_dict

    def naive_bayes(self, X_train, y_train):
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        model = GaussianNB()
        for scoring in scores_dict.keys():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            scores_dict[scoring].append(np.mean(scores))
        return scores_dict

    def random_forest(self, X_train, y_train):
        max_depth = [5, 10, 15]
        n_estimators = [100, 200, 300]
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for estimator in n_estimators:
                model = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42,)
                for scoring in scores_dict.keys():
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring].append(np.mean(scores))
        return scores_dict


ROOT = "./COVID19_data/"
DF_TRAIN = "extended_df_train_preprocessed.csv"
DF_VALIDATION = "extended_df_validation.csv"
TRANSFORMERS_FILE = "quantitative_transformers.pkl"


def get_validation_df():
    """
    This function will load the validation dataframe. Then, it will put it through the same fitted transformations
    of the training dataframe and return it.

    The transformations should be:
    1. Transformations of the quantitative variables.
    2. Imputation of missing values.

    Removal of outliers should not happen.
    As encoding of nominal variables is fixed, it should not be required for validation.

    :return: pd.DataFrame
    """
    # Load validation dataset
    validation_df = pd.read_csv(ROOT + DF_VALIDATION)

    # Load the saved transformers
    with open(TRANSFORMERS_FILE, "rb") as f:
        q_transformers = pickle.load(f)

    # Apply the transformers to the corresponding columns
    for var, transformer in q_transformers.items():
        mask = ~validation_df[var].isna()  # Mask for non-NaN values
        validation_df.loc[mask, var] = transformer.transform(
            validation_df.loc[mask, var].values.reshape(-1, 1)
        )

    # Load the saved standard scalers
    with open("scalers.pkl", "rb") as f:
        std_scalers = pickle.load(f)

    # Apply the scalers to the corresponding columns
    for var, scaler in std_scalers.items():
        mask = ~validation_df[var].isna()  # Mask for non-NaN values
        validation_df.loc[mask, var] = scaler.transform(
            validation_df.loc[mask, var].values.reshape(-1, 1)
        )


    # TODO fill the validation dataset with the exact same imputer of the train set


    return validation_df

if __name__ == '__main__':
    train_df = pd.read_csv(ROOT + DF_TRAIN)

    M = Models()
