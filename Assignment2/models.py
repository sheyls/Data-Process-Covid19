from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import wittgenstein as lw


class Models():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X, y):
        max_depth = []
        min_samples_split = []
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for min_split in min_samples_split:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=42))
                ])
                for scoring in scores_dict.keys():
                    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                    scores_dict[scoring].append(np.mean(scores))
                    print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

    def rule_induction(self, X, y):
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', lw.RIPPER())
        ])
        for scoring in scores_dict.keys():
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
            scores_dict[scoring].append(np.mean(scores))
            print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

    def logistic_regression(self, X, y):
        penalties = []
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for penalty in penalties:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(penalty=penalty, max_iter=1000, random_state=42))
            ])
            for scoring in scores_dict.keys():
                scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                scores_dict[scoring].append(np.mean(scores))
                print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

    def svm(self, X, y):
        kernels = []
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for kernel in kernels:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(kernel=kernel))
            ])
            for scoring in scores_dict.keys():
                scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                scores_dict[scoring].append(np.mean(scores))
                print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

    def naive_bayes(self, X, y):
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GaussianNB())
        ])
        for scoring in scores_dict.keys():
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
            scores_dict[scoring].append(np.mean(scores))
            print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

    def random_forest(self, X, y):
        max_depth = []
        n_estimators = []
        scores_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for depth in max_depth:
            for estimator in n_estimators:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42,))
                ])
                for scoring in scores_dict.keys():
                    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                    scores_dict[scoring].append(np.mean(scores))
                    print(f"{scoring}: {np.mean(scores)}")
        return scores_dict

if __name__ == '__main__':
    M = Models()
