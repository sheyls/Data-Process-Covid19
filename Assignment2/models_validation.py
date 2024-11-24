import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import wittgenstein as lw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelValidation():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X_train, y_train, X_validation, y_validation, best_model):
        best_max_depth = int(best_model['max_depth'])
        best_criterion = best_model['criterion']

        results = pd.DataFrame(columns=['max_depth', 'criterion', 'accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])
        best_model = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'max_depth': best_max_depth,
            'criterion': best_criterion,
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results

    def rule_induction(self, X_train, y_train, X_validation, y_validation, best_model):
        best_k = best_model['K']

        results = pd.DataFrame(columns=['K', 'accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])
        best_model = lw.RIPPER(k=int(best_k))
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'K': best_k,
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results

    def logistic_regression(self, X_train, y_train, X_validation, y_validation, best_model):
        best_penalty = best_model['penalty']

        results = pd.DataFrame(columns=['penalty', 'accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])
        best_model = LogisticRegression(penalty=best_penalty, max_iter=1000, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'penalty': best_penalty,
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results

    def svm(self, X_train, y_train, X_validation, y_validation, best_model):
        best_kernel = best_model['kernel']

        results = pd.DataFrame(columns=['kernel', 'accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])
        best_model = SVC(kernel=best_kernel)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'kernel': best_kernel,
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results

    def naive_bayes(self, X_train, y_train, X_validation, y_validation, best_model):
        results = pd.DataFrame(columns=['accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])

        best_model = GaussianNB()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results

    def random_forest(self, X_train, y_train, X_validation, y_validation, best_model):
        best_max_depth = int(best_model['max_depth'])
        best_n_estimators = int(best_model['n_estimators'])

        results = pd.DataFrame(columns=['max_depth', 'n_estimators', 'accuracy_validation', 'precision_validation', 'recall_validation', 'f1_validation'])
        best_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        results['max_depth'] = best_max_depth
        results['n_estimators'] = best_n_estimators
        new_row = {
            'max_depth': best_max_depth,
            'n_estimators': best_n_estimators,
            'accuracy_validation': accuracy_score(y_validation, y_pred),
            'precision_validation': precision_score(y_validation, y_pred, average='weighted'),
            'recall_validation': recall_score(y_validation, y_pred, average='weighted'),
            'f1_validation': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        return results