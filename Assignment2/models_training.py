import os
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
import numpy as np
import wittgenstein as lw
from models_validation import ModelValidation
from sklearn.model_selection import train_test_split
from skopt.space import Real, Categorical, Integer
import ast



from skopt import BayesSearchCV

class MixedBayes(ClassifierMixin, BaseEstimator):
    def __init__(self, binary_vars, quantitative_vars):
        self.bernoulli_nb = BernoulliNB()
        self.gaussian_nb = GaussianNB()
        self.binary_vars = binary_vars
        self.quantitative_vars = quantitative_vars

    def predict_proba(self, X):
        # Calculate probabilities from each model
        probs_binary = self.bernoulli_nb.predict_proba(X[self.binary_vars])
        probs_continuous = self.gaussian_nb.predict_proba(X[self.quantitative_vars])

        # Combine probabilities by multiplying
        combined_probs = probs_binary * probs_continuous

        # Divide by prior (same for both models)
        priors = self.bernoulli_nb.class_prior_
        adjusted_probs = combined_probs / priors

        # Normalize probabilities
        final_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
        return final_probs

    def predict(self, X):
        return self.predict_proba(X)

    def fit(self,X,y):
        # Train BernoulliNB on binary variables

        self.bernoulli_nb.fit(X[self.binary_vars], y)

        # Train GaussianNB on quantitative variables
        self.gaussian_nb.fit(X[self.quantitative_vars], y)


def nested_bayes_search(X, y, model, search_space, fss="univariate"):
    # Initialize the Bayesian optimizer
    new_search_space = {}
    for key, value in search_space.items():
        new_search_space["model__" + key] = value

    if fss == "univariate":
        pipeline = Pipeline([
            ("fss", SelectKBest(score_func=f_classif)),
            ("model", model)
        ])
        new_search_space["fss__k"] = Integer(4, X.shape[1])
    elif fss == "wrapper":
        pipeline = Pipeline([
            ("fss", SequentialFeatureSelector(model)),
            ("model", model)
        ])
    else:
        raise ValueError(f"FSS strategy {fss} is unimplemented.")

    bayes_search = BayesSearchCV(
        estimator=pipeline,  # Assuming RIPPER takes 'k' as a hyperparameter
        search_spaces=new_search_space,
        scoring="roc_auc",
        cv=2,  # Inner CV for hyperparameter tuning
        n_iter=2,  # Number of optimization iterations
        random_state=42
    )

    # Define the outer CV split
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform nested cross-validation
    outer_scores = []
    best_kwargs = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit the Bayesian optimizer on the inner CV
        bayes_search.fit(X_train, y_train)

        # Evaluate on the outer test set
        best_model = bayes_search.best_estimator_

        """
        All allowed scores:
        'neg_root_mean_squared_error', 'precision_weighted', 'roc_auc_ovr', 'recall_micro', 
        'f1_samples', 'neg_mean_poisson_deviance', 'neg_log_loss', 'r2', 'balanced_accuracy', 
        'recall_samples', 'recall_weighted', 'roc_auc_ovo_weighted', 'jaccard', 'precision_macro', 
        'neg_median_absolute_error', 'roc_auc', 'neg_negative_likelihood_ratio', 'f1_weighted', 
        'roc_auc_ovo', 'precision_micro', 'neg_mean_absolute_error', 'max_error', 'mutual_info_score', 
        'neg_mean_squared_error', 'adjusted_rand_score', 'f1_macro', 'matthews_corrcoef', 
        'adjusted_mutual_info_score', 'completeness_score', 'top_k_accuracy', 'neg_mean_absolute_percentage_error', 
        'recall', 'neg_mean_gamma_deviance', 'jaccard_micro', 'jaccard_weighted', 'average_precision', 'neg_brier_score', 
        'neg_mean_squared_log_error', 'rand_score', 'jaccard_macro', 'precision', 'jaccard_samples', 'recall_macro', 
        'f1_micro', 'positive_likelihood_ratio', 'fowlkes_mallows_score', 'homogeneity_score', 
        'neg_root_mean_squared_log_error', 'f1', 'v_measure_score', 'roc_auc_ovr_weighted', 
        'd2_absolute_error_score', 'precision_samples', 'normalized_mutual_info_score', 'explained_variance', 
        'accuracy'
        """
        test_score = cross_validate(best_model, X_test, y_test, cv=5, scoring=["f1", "roc_auc",'accuracy', 'precision', 'recall'])
        cleaned_scores = {}
        for key, value in test_score.items():
            cleaned_scores[key] = (np.mean(value), np.std(value))
        outer_scores.append(cleaned_scores)
        best_kwargs.append(bayes_search.best_params_)
    # Compile results
    result = pd.DataFrame(data={"scores": outer_scores, "hyperparams": best_kwargs})

    return result


class ModelTraining():
    def __init__(self) -> None:
        pass

    def bayes_decision_tree(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "criterion": Categorical(["gini", "entropy"]),
            "max_depth": Integer(5, 15)
        }
        return nested_bayes_search(X, y, DecisionTreeClassifier(random_state=32), search_space)

    def bayes_rule_induction(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "k": Integer(1, 8),
        }
        return nested_bayes_search(X, y, lw.RIPPER(random_state=32), search_space)

    def bayes_logistic_regression(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "penalties": Categorical([None, 'l2'])
        }
        return nested_bayes_search(X, y, LogisticRegression(random_state=32), search_space)

    def bayes_svm(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "kernel": Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            "C": Real(1e-20, 2)
        }
        return nested_bayes_search(X, y, SVC(random_state=32), search_space)

    def bayes_naive_bayes(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "var_smoothing" : Real(1e-20, 1e-5)
        }
        return nested_bayes_search(X, y, GaussianNB(), search_space)

    def bayes_random_forest(self, X, y):
        # Define the hyperparameter search space
        search_space = {
            "max_depth": Integer(5,  15),
            "n_estimators": Integer(100, 300)
        }
        return nested_bayes_search(X, y, RandomForestClassifier(random_state=32), search_space)

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
                    scores_dict[scoring] = np.mean(scores), np.std(scores)
                scores_dict['max_depth'] = depth
                scores_dict['criterion'] = criteria
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
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
                scores_dict[scoring] = np.mean(scores), np.std(scores)
            scores_dict['K'] = k
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
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
                scores_dict[scoring] = np.mean(scores), np.std(scores)
            scores_dict['penalty'] = penalty
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
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
                scores_dict[scoring] = np.mean(scores), np.std(scores)
            scores_dict['kernel'] = kernel
            results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
        return results, best_model

    def naive_bayes(self, X_train, y_train):
        results = pd.DataFrame()
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        model = GaussianNB()
        for scoring in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            scores_dict[scoring] = np.mean(scores), np.std(scores)
        results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
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
                    scores_dict[scoring] = np.mean(scores), np.std(scores)
                scores_dict['max_depth'] = depth
                scores_dict['n_estimators'] = estimator
                results = pd.concat([results, pd.DataFrame([scores_dict])], ignore_index=True)
        best_f1_score = results['f1'].idxmax()
        best_model = results.loc[best_f1_score]
        best_model['Evaluation'] = 'Train'
        return results, best_model


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
    #with open(TRANSFORMERS_FILE, "rb") as f:
    #    q_transformers = pickle.load(f)

    # Apply the transformers to the corresponding columns
    #for var, transformer in q_transformers.items():
    #    mask = ~validation_df[var].isna()  # Mask for non-NaN values
    #    validation_df.loc[mask, var] = transformer.transform(
    #        validation_df.loc[mask, var].values.reshape(-1, 1)
    #    )

    # Load the saved standard scalers
    with open("COVID19_data/scalers.pkl", "rb") as f:
        std_scalers = pickle.load(f)

    # Apply the scalers to the corresponding columns
    for var, scaler in std_scalers.items():
        mask = ~validation_df[var].isna()  # Mask for non-NaN values
        validation_df.loc[mask, var] = scaler.transform(
            validation_df.loc[mask, var].values.reshape(-1, 1)
        )


    # TODO fill the validation dataset with the exact same imputer of the train set


    return validation_df


ROOT = "./COVID19_data/"
DF_TRAIN = "extended_df_train_preprocessed_standard.csv"
DF_VALIDATION = "extended_df_validation.csv"
TRANSFORMERS_FILE = "quantitative_transformers.pkl"


def main():
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

    X_train = train_df[quantitative_vars + nominal_vars + ordinal_vars].astype(float)
    y_train = train_df[target_var]

    # Training
    MT = ModelTraining()
    models = [
        ("bayes_decision_tree", MT.bayes_decision_tree),
        ("bayes_rule_induction", MT.bayes_rule_induction),
        ("bayes_logistic_regression", MT.bayes_logistic_regression),
        ("bayes_naive_bayes", MT.bayes_naive_bayes),
        ("bayes_svm", MT.bayes_svm),
    ]
    for name, f in models:
        print(f"Training {name}")
        results = f(X_train, y_train)

        # Convertir los resultados en un dataframe
        rows = []
        for _, row in results.iterrows():
            scores = row['scores']
            hyperparams = row['hyperparams']
            row_data = {**hyperparams}
            for key, value in scores.items():
                if isinstance(value, tuple):
                    # Si el valor es una tupla, dividir en "mean" y "std"
                    row_data[f"{key}_mean"] = value[0]
                    row_data[f"{key}_std"] = value[1]
                else:
                    # Si no es una tupla, usar el valor directamente
                    row_data[key] = value
            rows.append(row_data)
        df_results = pd.DataFrame(rows)

        print(f"Saving the results of {name}")
        df_results.to_csv(os.path.join(ROOT, f"/Results/results_{name}.csv"), index=False)

if __name__ == '__main__':
    main()