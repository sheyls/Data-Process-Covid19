import os
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
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


def nested_bayes_search(X, y, model, search_space):
    # Initialize the Bayesian optimizer with lw.RIPPER
    bayes_search = BayesSearchCV(
        estimator=model,  # Assuming RIPPER takes 'k' as a hyperparameter
        search_spaces=search_space,
        scoring="roc_auc",
        cv=5,  # Inner CV for hyperparameter tuning
        n_iter=20,  # Number of optimization iterations
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
        test_score = cross_validate(best_model, X_test, y_test, cv=5, scoring=["f1", "roc_auc",'accuracy', 'precision', 'recall']).mean()
        outer_scores.append(test_score)
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
            "kernels": Categorical(['linear', 'poly', 'rbf', 'sigmoid'])
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
        print(f"Saving the results of {name}")
        results.to_csv(os.path.join(ROOT, f"results_{name}.csv"),index=False)

    """
    train_results_svm, best_svm = MT.svm(X_train, y_train)
    train_results_nb, best_nb = MT.naive_bayes(X_train, y_train)
    train_results_tree, best_tree = MT.decision_tree(X_train, y_train)
    train_results_rf, best_rf = MT.random_forest(X_train, y_train)
    train_results_log, best_log = MT.logistic_regression(X_train, y_train)
    train_results_rule, best_rule = MT.rule_induction(X_train, y_train)

    # Validating
    MV = ModelValidation()
    validation_df = get_validation_df()

    X_val = validation_df[quantitative_vars + nominal_vars + ordinal_vars]
    y_val = validation_df[target_var]

    val_results_svm = MV.svm(X_train, y_train, X_val, y_val, best_svm)
    val_results_nb = MV.naive_bayes(X_train, y_train, X_val, y_val, best_nb)
    val_results_tree = MV.decision_tree(X_train, y_train, X_val, y_val, best_tree)
    val_results_rf = MV.random_forest(X_train, y_train, X_val, y_val, best_rf)
    val_results_log = MV.logistic_regression(X_train, y_train, X_val, y_val, best_log)
    val_results_rule = MV.rule_induction(X_train, y_train, X_val, y_val, best_rule)

    # Appending training metrics with validation metrics
    results_svm = pd.concat([best_svm, val_results_svm], axis=1)
    results_nb = pd.concat([best_nb, val_results_nb], axis=1)
    results_tree = pd.concat([best_tree, val_results_tree], axis=1)
    results_rf = pd.concat([best_rf, val_results_rf], axis=1)
    results_log = pd.concat([best_log, val_results_log], axis=1)
    results_rule = pd.concat([best_rule, val_results_rule], axis=1)

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
    """