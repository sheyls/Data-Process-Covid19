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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Models():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X_train, y_train, X_validation, y_validation):
        criterion = ['gini', 'entropy']
        max_depth = [5, 10, 15]
        results = pd.DataFrame()
        scores_dict = {'max_depth': [], 'criterion': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        for depth in max_depth:
            for criteria in criterion:
                model = DecisionTreeClassifier(criterion=criteria, max_depth=depth, random_state=42)
                for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring] = np.mean(scores)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_validation)
                scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
                scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
                scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
                scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
                scores_dict['max_depth'] = depth
                scores_dict['criterion'] = criteria
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

    def rule_induction(self, X_train, y_train, X_validation, y_validation):
        K = [1, 2, 3, 4]
        results = pd.DataFrame()
        scores_dict = {'K': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        for k in K:
            model = lw.RIPPER(k=k)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_validation)
            scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
            scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
            scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
            scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
            scores_dict['K'] = k
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

    def logistic_regression(self, X_train, y_train, X_validation, y_validation):
        penalties = [None, 'l2']
        results = pd.DataFrame()
        scores_dict = {'penalty': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        for penalty in penalties:
            model = LogisticRegression(penalty=penalty, max_iter=1000, random_state=42)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_validation)
            scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
            scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
            scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
            scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
            scores_dict['penalty'] = penalty
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

    def svm(self, X_train, y_train, X_validation, y_validation):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = pd.DataFrame()
        scores_dict = {'kernel': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        for kernel in kernels:
            model = SVC(kernel=kernel)
            for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores_dict[scoring] = np.mean(scores)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_validation)
            scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
            scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
            scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
            scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
            scores_dict['kernel'] = kernel
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

    def naive_bayes(self, X_train, y_train, X_validation, y_validation):
        results = pd.DataFrame()
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        model = GaussianNB()
        for scoring in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            scores_dict[scoring] = np.mean(scores)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation)
        scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
        scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
        scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
        scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
        results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

    def random_forest(self, X_train, y_train, X_validation, y_validation):
        max_depth = [5, 10, 15]
        n_estimators = [100, 200, 300]
        results = pd.DataFrame()
        scores_dict = {'max_depth': [], 'n_estimators': [],
                       'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                       'accuracy_validation': [], 'precision_validation': [], 'recall_validation': [], 'f1_validation': []}
        for depth in max_depth:
            for estimator in n_estimators:
                model = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42,)
                for scoring in ['accuracy', 'precision', 'recall', 'f1']:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    scores_dict[scoring] = np.mean(scores)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_validation)
                scores_dict['accuracy_validation'] = accuracy_score(y_validation, y_pred)
                scores_dict['precision_validation'] = precision_score(y_validation, y_pred, average='weighted')
                scores_dict['recall_validation'] = recall_score(y_validation, y_pred, average='weighted')
                scores_dict['f1_validation'] = f1_score(y_validation, y_pred, average='weighted')
                scores_dict['max_depth'] = depth
                scores_dict['n_estimators'] = estimator
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        return results

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
    train_df = pd.read_csv(ROOT + DF_TRAIN)
    M = Models()

    validation_df = get_validation_df()