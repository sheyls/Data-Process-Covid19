import os
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
    train_df = pd.read_csv(ROOT + DF_TRAIN)

    k = len("country_of_residence")
    countries = [col_name for col_name in list(train_df.columns) if col_name[:k] == "country_of_residence"]

    nominal_vars = [
        "sex", "history_of_fever", "cough", "sore_throat",
        "runny_nose", "wheezing", "shortness_of_breath", "headache", "loss_of_taste",
        "fatigue_malaise", "muscle_aches", "joint_pain", "diarrhoea", "vomiting_nausea",
        "chronic_cardiac_disease", "hypertension", "chronic_pulmonary_disease", "asthma", "smoking"
    ] + countries

    ordinal_vars = [
        "date_of_first_symptoms_month", "date_of_first_symptoms_dayofyear",
        # "admission_date_month", "admission_date_dayofyear"
    ]

    quantitative_vars = [
        "age", "fever_temperature", "oxygen_saturation",
        "date_of_first_symptoms_sin", "date_of_first_symptoms_cos",
        # "admission_date_sin", "admission_date_cos"
    ]

    target_var = "PCR_result"

    X_train = train_df[quantitative_vars + nominal_vars + ordinal_vars]
    y_train = train_df[target_var]

    MT = ModelTraining()
    best_svm, results_svm = MT.svm(X_train, y_train)
    best_nb, results_nb = MT.naive_bayes(X_train, y_train)
    best_tree, results_tree = MT.decision_tree(X_train, y_train)
    best_rf, results_rf = MT.random_forest(X_train, y_train)
    best_log, results_log = MT.logistic_regression(X_train, y_train)
    best_rule, results_rule = MT.rule_induction(X_train, y_train)

    # Saving to files
    filepaths = {
        'best_svm': os.path.join(ROOT + "outputs/best_svm.csv"),
        'results_svm': os.path.join(ROOT + "outputs/results_svm.csv"),
        'best_nb': os.path.join(ROOT + "outputs/best_nb.csv"),
        'results_nb': os.path.join(ROOT + "outputs/results_nb.csv"),
        'best_tree': os.path.join(ROOT + "outputs/best_tree.csv"),
        'results_tree': os.path.join(ROOT + "outputs/results_tree.csv"),
        'best_rf': os.path.join(ROOT + "outputs/best_rf.csv"),
        'results_rf': os.path.join(ROOT + "outputs/results_rf.csv"),
        'best_log': os.path.join(ROOT + "outputs/best_log.csv"),
        'results_log': os.path.join(ROOT + "outputs/results_log.csv"),
        'best_rule': os.path.join(ROOT + "outputs/best_rule.csv"),
        'results_rule': os.path.join(ROOT + "outputs/results_rule.csv")
    }

    dfs = {
        'best_svm': best_svm,
        'results_svm': results_svm,
        'best_nb': best_nb,
        'results_nb': results_nb,
        'best_tree': best_tree,
        'results_tree': results_tree,
        'best_rf': best_rf,
        'results_rf': results_rf,
        'best_log': best_log,
        'results_log': results_log,
        'best_rule': best_rule,
        'results_rule': results_rule
    }

    for name, df in dfs.items():
        df.to_csv(filepaths[name], index=False)

    MV = ModelValidation()
    validation_df = get_validation_df()

    X_val = validation_df[quantitative_vars + nominal_vars + ordinal_vars]
    y_val = validation_df[target_var]

    results_svm = MV.svm(X_train, y_train, X_val, y_val, best_svm)
    results_nb = MV.naive_bayes(X_train, y_train, X_val, y_val, best_nb)
    results_tree = MV.decision_tree(X_train, y_train, X_val, y_val, best_tree)
    results_rf = MV.random_forest(X_train, y_train, X_val, y_val, best_rf)
    results_log = MV.logistic_regression(X_train, y_train, X_val, y_val, best_log)
    results_rule = MV.rule_induction(X_train, y_train, X_val, y_val, best_rule)

    filepaths = {
        'results_svm': os.path.join(ROOT + "outputs/results_svm_val.csv"),
        'results_nb': os.path.join(ROOT + "outputs/results_nb_val.csv"),
        'results_tree': os.path.join(ROOT + "outputs/results_tree_val.csv"),
        'results_rf': os.path.join(ROOT + "outputs/results_rf_val.csv"),
        'results_log': os.path.join(ROOT + "outputs/results_log_val.csv"),
        'results_rule': os.path.join(ROOT + "outputs/results_rule_val.csv")
    }

    dfs = {
        'results_svm': results_svm,
        'results_nb': results_nb,
        'results_tree': results_tree,
        'results_rf': results_rf,
        'results_log': results_log,
        'results_rule': results_rule
    }

    for name, df in dfs.items():
        df.to_csv(filepaths[name], index=False)