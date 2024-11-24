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
from models_validation import ModelValidation
from sklearn.model_selection import train_test_split

class ModelTraining():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X_train, y_train):
        criterion = ['gini', 'entropy']
        max_depth = [5, 10, 15]
        results = pd.DataFrame()
        scores_dict = {'max_depth': [], 'criterion': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for criteria in criterion:
                model = DecisionTreeClassifier(criterion=criteria, max_depth=depth, random_state=42)
                for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring] = np.mean(scores)
                scores_dict['max_depth'] = depth
                scores_dict['criterion'] = criteria
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def rule_induction(self, X_train, y_train):
        K = [1, 2, 3, 4]
        results = pd.DataFrame()
        scores_dict = {'K': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for k in K:
            model = lw.RIPPER(k=k)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            scores_dict['K'] = k
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def logistic_regression(self, X_train, y_train):
        penalties = [None, 'l2']
        results = pd.DataFrame()
        scores_dict = {'penalty': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for penalty in penalties:
            model = LogisticRegression(penalty=penalty, max_iter=1000, random_state=42)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            scores_dict['penalty'] = penalty
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def svm(self, X_train, y_train):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = pd.DataFrame()
        scores_dict = {'kernel': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for kernel in kernels:
            model = SVC(kernel=kernel)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            scores_dict['kernel'] = kernel
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def naive_bayes(self, X_train, y_train):
        results = pd.DataFrame()
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        model = GaussianNB()
        for scoring in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            scores_dict[scoring] = np.mean(scores)
        results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def random_forest(self, X_train, y_train):
        max_depth = [5, 10, 15]
        n_estimators = [100, 200, 300]
        results = pd.DataFrame()
        scores_dict = {'max_depth': [], 'n_estimators': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for estimator in n_estimators:
                model = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42)
                for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring] = np.mean(scores)
                scores_dict['max_depth'] = depth
                scores_dict['n_estimators'] = estimator
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        return results, best_model

    def separating_target(self, data):
        X = data.drop(columns=['PAID_ON_TIME'])
        y = data['PAID_ON_TIME']
        return X, y

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
    # train_df = pd.read_csv(ROOT + DF_TRAIN)
    MT = ModelTraining()
    MV = ModelValidation()
    encoded_data = pd.read_csv('/Users/mariajosefranco/Desktop/EnerBit/repos/ML/encoded_enerbit_data.csv')
    encoded_data = encoded_data.drop(
        columns=[
            "SERVICE_TYPE",
            "COUNTRY",
            "TOTAL_BILLED_AMOUNT_VALIDATION",
            "DAYS_PAST_DUE",
            "YEAR_PAYMENT_DATE",
            "MONTH_PAYMENT_DATE",
            "DAY_PAYMENT_DATE",
        ]
    )
    encoded_data = encoded_data[:500]
    X, y = MT.separating_target(encoded_data)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    train_results, best_model = MT.random_forest(X_train, y_train)
    validation_results = MV.random_forest(X_train, y_train, X_validation, y_validation, best_model)
    train_results
    # validation_df = get_validation_df()