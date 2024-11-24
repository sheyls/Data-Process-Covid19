from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import shap
import wittgenstein as lw
import seaborn as sns


def interpret_tree(model : DecisionTreeClassifier, feature_names=None, class_names=None, X_test=None, y_test=None):
    # 1. Decision Tree Visualization
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Decision Tree Visualization")
    plt.show()

    # 2. Feature Importances
    feature_importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, feature_importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Feature Importances in Decision Tree")
    plt.show()

    # 3. Path Explanation for a Prediction
    sample_index = 0  # Change to any test sample
    sample = X_test[sample_index].reshape(1, -1)
    decision_path = model.decision_path(sample)

    # Display the decision rules
    rules = export_text(model, feature_names=feature_names)
    print("Decision Rules:\n", rules)

    # Trace the prediction path
    node_indicator = decision_path.indices
    print(f"\nDecision Path for Sample {sample_index}:")
    for node_id in node_indicator:
        print(f"Rule {node_id}: {rules.splitlines()[node_id]}")

    # 4. SHAP Summary Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot (global interpretation)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    # SHAP force plot (local interpretation for one prediction)
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][sample_index],
        X_test[sample_index],
        feature_names=feature_names
    )

    for target in np.unique(y_test):
        plot_partial_dependence(model, X_test, features=range(len(feature_names)), feature_names=feature_names,
                                grid_resolution=50, target=target)
        plt.suptitle('Partial Dependence Plots (PDP)')
        plt.show()


def interpret_lw(model : lw.RIPPER, class_names=None, feature_names=None, X_test=None, y_test=None):
    # 1. Display Learned Rules
    print("Learned Rules:")
    for rule in model.rules_:
        print(rule)

    # 2. Feature Importance from Rule Frequency
    feature_counts = {feature: 0 for feature in feature_names}
    for rule in model.rules_:
        for condition in rule.conditions:
            feature = condition.feature
            feature_counts[feature_names[feature]] += 1

    feature_importance_df = pd.DataFrame(
        {'Feature': list(feature_counts.keys()), 'Rule Frequency': list(feature_counts.values())}
    ).sort_values(by='Rule Frequency', ascending=False)

    print("\nFeature Importance (by Rule Frequency):")
    print(feature_importance_df)

    # Visualization of Feature Importance
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Rule Frequency'], color='skyblue')
    plt.xlabel("Rule Frequency")
    plt.title("Feature Importance (by Rule Frequency in RIPPER Rules)")
    plt.show()

    for target in np.unique(y_test):
        plot_partial_dependence(model, X_test, features=range(len(feature_names)), feature_names=feature_names,
                                grid_resolution=50, target=target)
        plt.suptitle('Partial Dependence Plots (PDP)')
        plt.show()

    # 3. Example-Based Explanation
    sample_index = 0  # Choose any sample from the test set
    sample = X_test[sample_index].reshape(1, -1)
    predicted_class = model.predict(sample)[0]

    # Find matching rule
    matching_rule = None
    for rule in model.rules_:
        if rule.evaluate(sample):
            matching_rule = rule
            break

    print(f"\nExplanation for Test Sample {sample_index} (Predicted Class = {class_names[predicted_class]}):")
    if matching_rule:
        print(f"Matched Rule: {matching_rule}")
    else:
        print("No matching rule found.")


def interpret_gaussianNB(model: GaussianNB, class_names=None, feature_names=None, X_test=None, y_test=None):
    # Ensure that feature names and class names are provided
    if feature_names is None or class_names is None:
        raise ValueError("Both 'feature_names' and 'class_names' must be provided.")

    # 1. Class Probabilities
    # For each test sample, display the predicted probabilities for each class
    sample_index = 0  # Choose any sample from the test set
    sample = X_test[sample_index].reshape(1, -1)

    # Get predicted probabilities for all classes
    prob = model.predict_proba(sample)[0]
    predicted_class = model.predict(sample)[0]

    print(
        f"\nExample-based Explanation for Test Sample {sample_index} (Predicted Class = {class_names[predicted_class]}):")
    print(f"Prediction Probability for Each Class: {prob}")

    # 2. Feature Distributions (Class-conditional distributions)
    print("\nFeature Distributions for Each Class:")
    means = model.theta_  # Mean of the features for each class
    variances = model.sigma_  # Variance of the features for each class

    for i, feature in enumerate(feature_names):
        print(f"\nFeature: {feature}")
        for j, class_name in enumerate(class_names):
            print(f"Class {class_name}: Mean = {means[j, i]:.4f}, Variance = {variances[j, i]:.4f}")

            # Visualize feature distribution for this class
            x_vals = np.linspace(means[j, i] - 4 * np.sqrt(variances[j, i]), means[j, i] + 4 * np.sqrt(variances[j, i]),
                                 100)
            y_vals = (1 / (np.sqrt(2 * np.pi * variances[j, i]))) * np.exp(
                -0.5 * ((x_vals - means[j, i]) ** 2) / variances[j, i])
            plt.plot(x_vals, y_vals, label=f"Class {class_name} (Feature: {feature})")

        plt.title("Class Conditional Distributions for Each Feature")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


    for target in np.unique(y_test):
        plot_partial_dependence(model, X_test, features=range(len(feature_names)), feature_names=feature_names,
                                grid_resolution=50, target=target)
        plt.suptitle('Partial Dependence Plots (PDP)')
        plt.show()

    # 3. Error Analysis
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def interpret_logistic_regression(model: LogisticRegression, class_names=None, feature_names=None, X_test=None, y_test=None):
    # Ensure that feature names and class names are provided
    if feature_names is None or class_names is None:
        raise ValueError("Both 'feature_names' and 'class_names' must be provided.")

    # 1. Coefficients of the Logistic Regression Model
    coefficients = model.coef_[0]
    print("\nModel Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef:.4f}")

    # 2. Feature Importance from Coefficients
    feature_importance_df = pd.DataFrame(
        {'Feature': feature_names, 'Coefficient': coefficients, 'Abs_Coefficient': np.abs(coefficients)}
    ).sort_values(by='Abs_Coefficient', ascending=False)

    print("\nFeature Importance (based on Coefficients):")
    print(feature_importance_df)

    # Visualization of Feature Importance (Bar plot of absolute coefficients)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Abs_Coefficient'], color='skyblue')
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Feature Importance in Logistic Regression")
    plt.show()

    # 3. Example-Based Explanation
    sample_index = 0  # Choose any sample from the test set
    sample = X_test[sample_index].reshape(1, -1)

    # Calculate prediction probability
    prob = model.predict_proba(sample)[0]
    predicted_class = model.predict(sample)[0]

    print(
        f"\nExample-based Explanation for Test Sample {sample_index} (Predicted Class = {class_names[predicted_class]}):")

    # Compute the linear decision function (z) and its contribution by feature
    linear_combination = np.dot(sample, coefficients)
    print(f"Linear combination (z): {linear_combination[0]:.4f}")

    for i, feature in enumerate(feature_names):
        contribution = sample[0, i] * coefficients[i]
        print(f"Contribution of {feature}: {contribution:.4f}")

    print(f"Prediction Probability (Class {class_names[predicted_class]}): {prob[predicted_class]:.4f}")

    # 4. Error Analysis
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def interpret_svm(model: SVC, class_names=None, feature_names=None, X_test=None, y_test=None, X_train=None):
    # Ensure that feature names and class names are provided
    if feature_names is None or class_names is None:
        raise ValueError("Both 'feature_names' and 'class_names' must be provided.")

    # 1. Support Vectors
    support_vectors = model.support_  # Indices of support vectors
    print(f"\nSupport Vectors Indices: {support_vectors}")

    # Support Vector data points
    support_vectors_data = X_train[support_vectors]
    print(f"\nSupport Vectors Data Points:\n{support_vectors_data}")

    # 2. Coefficients (Weights) for Linear Kernel
    if model.kernel == 'linear':
        coef = model.coef_[0]
        print("\nCoefficients (weights) for the Linear Kernel:")
        for feature, weight in zip(feature_names, coef):
            print(f"{feature}: {weight:.4f}")

        # Visualize the decision boundary
        plot_decision_boundary(X_test, y_test, model, feature_names)

    # 3. Decision Function (Confidence for Prediction)
    sample_index = 0  # Choose any sample from the test set
    sample = X_test[sample_index].reshape(1, -1)

    # Get the decision function score (distance to the hyperplane)
    decision_value = model.decision_function(sample)
    print(f"\nDecision Function for Test Sample {sample_index} (Distance to Hyperplane): {decision_value[0]}")

    # 4. Example-Based Explanation
    predicted_class = model.predict(sample)[0]
    print(
        f"\nExample-based Explanation for Test Sample {sample_index} (Predicted Class = {class_names[predicted_class]}):")
    print(f"Decision Function Value: {decision_value[0]}")

    # 5. Error Analysis
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Step 1: Create combinations of feature pairs
    num_features = X_train.shape[1]
    feature_combinations = combinations(range(num_features), 2)

    # Step 2: Loop over all pairs of features and plot decision boundaries
    for pair in feature_combinations:
        feature1, feature2 = pair

        # Select the two features for this pair
        X_pair = X_train[:, [feature1, feature2]]

        # Train the SVM model on the selected pair of features
        model = SVC(kernel='linear')
        model.fit(X_pair, y_train)

        # Create the meshgrid for the decision boundary plot
        x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
        y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # Predict on the meshgrid points
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

        # Plot the data points for the selected pair of features
        plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=50)

        # Labels and title
        plt.title(f"SVM Decision Boundary: Feature {feature1 + 1} vs Feature {feature2 + 1}")
        plt.xlabel(f"Feature {feature1 + 1}")
        plt.ylabel(f"Feature {feature2 + 1}")
        plt.colorbar()  # Show class color legend

        # Show the plot
        plt.show()




def plot_decision_boundary(X, y, model, feature_names):
    # For visualization of decision boundary for 2D data
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.75)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)

    plt.title("SVM Decision Boundary (Linear Kernel)")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()


def interpret_rfc(model: RandomForestClassifier, X_test, y_test, class_names=None, feature_names=None):
    # 1. Feature Importances
    feature_importances = model.feature_importances_
    print("\nFeature Importances:")
    for feature, importance in zip(feature_names, feature_importances):
        print(f"{feature}: {importance:.4f}")

    # Plot Feature Importances
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()

    # 2. Visualizing One of the Trees
    tree = model.estimators_[0]  # Get the first tree in the Random Forest
    tree_rules = export_text(tree, feature_names=feature_names)
    print("\nRules of First Decision Tree in the Random Forest:")
    print(tree_rules)

    # 3. Partial Dependence Plots (PDP)
    for target in np.unique(y_test):
        plot_partial_dependence(model, X_test, features=range(len(feature_names)), feature_names=feature_names,
                                grid_resolution=50, target=target)
        plt.suptitle('Partial Dependence Plots (PDP)')
        plt.show()

    # 4. Performance Metrics
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()



if __name__ == "__main__":
    # Load dataset (For example, Iris dataset)
    data = load_iris()
    X = data.data
    y = data.target
    class_names = data.target_names
    feature_names = data.feature_names

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the models
    dt_model = DecisionTreeClassifier(random_state=42)
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gnb_model = GaussianNB()

    # Train the models
    dt_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gnb_model.fit(X_train, y_train)

    # Call interpretability functions for each model
    print("Interpretation for Decision Tree Classifier:")
    interpret_tree(dt_model, feature_names=feature_names, class_names=class_names, X_test=X_test, y_test=y_test)

    print("\nInterpretation for Logistic Regression:")
    interpret_logistic_regression(lr_model, class_names=class_names, feature_names=feature_names, X_test=X_test,
                                  y_test=y_test)

    print("\nInterpretation for Support Vector Machine:")
    interpret_svm(svm_model, class_names=class_names, feature_names=feature_names, X_test=X_test, y_test=y_test, X_train=X_train)

    print("\nInterpretation for Gaussian Naive Bayes:")
    interpret_gaussianNB(gnb_model, class_names=class_names, feature_names=feature_names, X_test=X_test, y_test=y_test)

    print("\nInterpretation for Random Forest Classifier:")
    interpret_rfc(rf_model, X_test=X_test, y_test=y_test, class_names=class_names, feature_names=feature_names)
