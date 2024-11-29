import os
import pickle

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import wittgenstein as lw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from wittgenstein import RIPPER
import dtreeviz  # remember to load the package

ROOT = "./COVID19_data/"
DF_TRAIN = "extended_df_train_preprocessed_standard.csv"
DF_VALIDATION = "extended_df_validation.csv"
TRANSFORMERS_FILE = "quantitative_transformers.pkl"


class ModelValidation():
    def __init__(self) -> None:
        pass

    def decision_tree(self, X_train, y_train, X_validation, y_validation, best_model):
        best_max_depth = int(best_model['max_depth'])
        best_criterion = best_model['criterion']

        results = pd.DataFrame(columns=['max_depth', 'criterion', 'accuracy', 'precision', 'recall', 'f1'])
        best_model = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'max_depth': best_max_depth,
            'criterion': best_criterion,
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results

    def rule_induction(self, X_train, y_train, X_validation, y_validation, best_model):
        best_k = best_model['K']

        results = pd.DataFrame(columns=['K', 'accuracy', 'precision', 'recall', 'f1'])
        best_model = lw.RIPPER(k=int(best_k))
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'K': best_k,
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results

    def logistic_regression(self, X_train, y_train, X_validation, y_validation, best_model):
        best_penalty = best_model['penalty']

        results = pd.DataFrame(columns=['penalty', 'accuracy', 'precision', 'recall', 'f1'])
        best_model = LogisticRegression(penalty=best_penalty, max_iter=1000, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'penalty': best_penalty,
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results

    def svm(self, X_train, y_train, X_validation, y_validation, best_model):
        best_kernel = best_model['kernel']

        results = pd.DataFrame(columns=['kernel', 'accuracy', 'precision', 'recall', 'f1'])
        best_model = SVC(kernel=best_kernel)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'kernel': best_kernel,
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results

    def naive_bayes(self, X_train, y_train, X_validation, y_validation, best_model):
        results = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])

        best_model = GaussianNB()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        new_row = {
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results

    def random_forest(self, X_train, y_train, X_validation, y_validation, best_model):
        best_max_depth = int(best_model['max_depth'])
        best_n_estimators = int(best_model['n_estimators'])

        results = pd.DataFrame(columns=['max_depth', 'n_estimators', 'accuracy', 'precision', 'recall', 'f1'])
        best_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_validation)
        results['max_depth'] = best_max_depth
        results['n_estimators'] = best_n_estimators
        new_row = {
            'max_depth': best_max_depth,
            'n_estimators': best_n_estimators,
            'accuracy': accuracy_score(y_validation, y_pred),
            'precision': precision_score(y_validation, y_pred, average='weighted'),
            'recall': recall_score(y_validation, y_pred, average='weighted'),
            'f1': f1_score(y_validation, y_pred, average='weighted')
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results['Evaluation'] = 'Validation'
        return results


def get_validation_df(scaled=True):
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

    # Load the saved transformers. This was deprecated
    #with open(TRANSFORMERS_FILE, "rb") as f:
    #    q_transformers = pickle.load(f)

    # Apply the transformers to the corresponding columns
    #for var, transformer in q_transformers.items():
    #    mask = ~validation_df[var].isna()  # Mask for non-NaN values
    #    validation_df.loc[mask, var] = transformer.transform(
    #        validation_df.loc[mask, var].values.reshape(-1, 1)
    #    )

    if scaled:
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
        "age",
        # "fever_temperature",
        "oxygen_saturation",
        "date_of_first_symptoms_sin", "date_of_first_symptoms_cos",
        # "admission_date_sin", "admission_date_cos"
    ]

    target_var = "PCR_result"

    X_train = train_df[quantitative_vars + nominal_vars + ordinal_vars].astype(float)
    y_train = train_df[target_var]

    validation_df = get_validation_df().dropna()
    X_val = validation_df[quantitative_vars + nominal_vars + ordinal_vars].astype(float)
    y_validation = validation_df[target_var]

    best_kwargs = {
        "fss__k": 25,
        "model__criterion": "entropy",
        "model__max_depth": 9
    }
    model = DecisionTreeClassifier(random_state=42)

    # best_kwargs = {
    #     "fss__k": 25,
    #     "model__max_depth": 15,
    #     # "model__max_features": "sqrt",
    #     # "model__min_samples_leaf": 1,
    #     # "model__min_samples_split": 2,
    #     # "model__criterion": "entropy",
    #     "model__n_estimators": 300
    # }
    # model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ("fss", SelectKBest(score_func=f_classif)),
        ("model", model)
    ])
    pipeline.set_params(**best_kwargs)

    pipeline.fit(X_train, y_train)
    y_predictions = pipeline.predict(X_val)

    # Accuracy
    accuracy = accuracy_score(y_validation, y_predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Precision
    precision = precision_score(y_validation, y_predictions, average='weighted')
    print(f"Precision: {precision:.2f}")

    # Recall
    recall = recall_score(y_validation, y_predictions, average='weighted')
    print(f"Recall: {recall:.2f}")

    # F1 Score
    f1 = f1_score(y_validation, y_predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_validation, y_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report (includes precision, recall, F1-score, and support)
    report = classification_report(y_validation, y_predictions)
    print("Classification Report:")
    print(report)

    # ROC-AUC (only if your model outputs probabilities and the task is binary classification)
    # Note: `y_prob` must be the probabilities for the positive class
    # roc_auc = roc_auc_score(y_validation, y_prob)

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    if isinstance(model, RandomForestClassifier):
        # Plot the first 3 trees in the Random Forest
        for i in range(3):
            plt.figure(figsize=(20, 10))
            plot_tree(model.estimators_[i],
                      feature_names=list(X_val.columns),
                      class_names=["Positive", "Negative"],
                      filled=True,
                      rounded=True)
            plt.title(f"Tree {i + 1} from Random Forest")
            plt.show()
    elif isinstance(model, DecisionTreeClassifier):
        selected_indices = pipeline.named_steps["fss"].get_support(indices=True)
        selected_columns = X_train.columns[selected_indices]

        viz_model = dtreeviz.model(model, X_train[selected_columns], y_train,
                                   target_name="target",
                                   feature_names=list(selected_columns),
                                   class_names=["Positive", "Negative"])

        viewer = viz_model.view(scale=0.8, orientation="LR", colors={
            "Positive": "lightblue",  # Color for class 0
            "Negative": "lightgreen"  # Color for class 1
        }, depth_range_to_display=(0, 4))
        viewer.show()
        viewer.save(os.path.join(ROOT, "decision_tree_dtreeviz.svg"))
        # Plot the first 3 trees in the Random Forest
        plt.figure(figsize=(20, 10), dpi=400)
        plot_tree(model,
                  feature_names=list(selected_columns),
                  class_names=["Positive", "Negative"],
                  filled=True,
                  impurity=False,
                  proportion=True,
                  # rounded=True,
                  max_depth=4,
                  fontsize=6  # Increase font size for larger cells
                  )
        plt.title(f"Decision Tree")
        plt.subplots_adjust()
        plt.savefig(os.path.join(ROOT, "decision_tree.png"))
        plt.show()

        # DOT data

    elif isinstance(model, RIPPER):
        # Extract rules as a list
        rules = model.ruleset_.rules

        # Format and display the rules
        print("\nFormatted Rules:")
        for i, rule in enumerate(rules):
            print(f"Rule {i + 1}: IF {rule.condition} THEN target = {rule.conclusion}")


if __name__ == '__main__':
    main()
