**Project Plan Template for Predicting Severe COVID-19 Cases**


## 1. Introduction

- **Context of the Project**: [Add brief explanation about COVID-19 inpatients and predicting severe outcomes.]
- **Problem Statement**: [Add details on identifying patients at risk of severe illness to enable early interventions.]
- **Stakeholders**: [List the stakeholders who will benefit, e.g., hospitals, healthcare organizations, policymakers.]

## 2. Cost/Benefit Analysis for Stakeholders

- **Benefits**: [Add information on how this predictive model will help stakeholders, including improvements in healthcare resource allocation, reduction in ICU admissions, better patient care, and financial savings.]
- **Costs**: [Detail the costs of developing and implementing the model, including technical resources, staff training, and maintenance.]
- **Return on Investment (ROI)**: [Estimate the potential return from preventing severe cases and avoiding costs.]

## 3. Framing the Problem as a Data Science Task

- **Objective**: [Define the objective of creating a predictive model to forecast severe illness based on demographic, clinical, and hospital admission data.]
- **Data**: [Identify data types and sources to be used, e.g., demographic data, clinical data, ICU admission, and outcomes.]
- **Approach**: [Outline machine learning techniques to consider, e.g., logistic regression, random forest, deep learning models, data handling methods, and evaluation metrics.]

### Objective

Our objective is to forecast severe COVID-19 illnesses among diabetes patients aged 40 to 6 in a reproducible and interpretable manner. 
Severe COVID-19 cases are defined based on these specific clinical criteria:
1. The need for intubation.
2. Requirement for artificial ventilation.
3. Hospitalization.
4. Admission to an Intensive Care Unit (ICU).


The aim is to predict these outcomes using extensive data currently accessible by the government, including:
1. Demographic data: Age, gender, ethnicity, and socioeconomic factors.
2. Clinical data: Information on comorbidities, diabetes management, and past medical history.
3. Hospital admission data: Admission status, current health metrics, and relevant treatment details.
4. Social and Behavioral Determinants of Health (SBDH): Factors like access to healthcare access, housing stability, and social support.

This data will go through preliminary pre-processing techniques in order to satisfy quality standards that assure that the model's training data is an accurate representation of the population. This will include steps such as cleaning the data and analyzing it to ensure that the demographical makeup of the data matches the target population to ensure reproducibility.

During modeling, interpretability will be assured through the prioritization of simple transparent models and the utilization model agnostic
tools such as SHAP (Shapley Additive Explanations) and Counterfactual Explanations to provide explanations on the relationships between the inputs and the results. This will allow us to ensure that the predictive model is comprehensible by clinicians and healthcare professionals, as they must be capable of understanding the model's predictions to make informed, life-altering decisions for the patients. Similarly, proving the replicability of the results across different hospitals and regions will allow healthcare professionals to trust that the model is robust and reliable.

Since our main objective is to reduce hospital resource usage and false positives have a low resource cost, the evaluation techniques to be used will be mainly sensitivity (true positive rate) and metrics like the F2 score over other evaluation metrics such as precision (positive predictive value).

### Data
#### Data provided
As described in the objective section, our data will be sourced from:
1. Demographic data.
2. Clinical data.
3. Hospital admission data.
4. Social and Behavioral Determinants of Health (SBDH).

In this section, we will delve into:
1. the actual data we expect to get from each source
1. the integrability
1. ethical concerns
1. completeness
1. data cleanliness and required reconciliation

#### Representative data (remove the ordinals)
In order to assure that the model's results will be replicable in practice, we will require that the data used is representative of the target population that will be predicted.

Firstly, each individual sampled should belong to the demographic subset we aim to predict, i.e. patients that have been diagnosed with diabetes and are in aged 40 to 60.

Secondly, the experimental distribution of the features used should follow their actual distribution of these features in our target population.

Thirdly, the sample size should be adequate to train reliable models, which means large enough to capture possibly critical subgroups that would otherwise appear as outliers in the preprocessing of the data.

Fourthly, the data should have balanced class distribution.

Fifthly, the data should be sampled from time periods and geographical locations that are relevant to the target population. In our case, we expect the individuals sampled to represent the geographical density distributions of the territory of Spain, and be from the last year to correctly represent the rapidly changing nature of COVID-19.

Sixthly, the data should be independent, without duplicates and avoiding highly correlated cases resulting from sampling within family units.

### Approach
In this project, we will consider a range of machine learning techniques tailored to the goal of predicting severe COVID-19 outcomes among diabetes patients. The primary focus will be on balancing predictive power with interpretability to ensure the model’s results can be understood and trusted by healthcare professionals. Below is an outline of the techniques and methods that will be used.

#### Data Pre-processing
In this phase, we will ensure the quality and consistency of the data before modeling. 
First, missing data will be handled using appropriate imputation techniques, such as mean or mode imputation for basic patterns, 
while more sophisticated methods like K-Nearest Neighbors (KNN) imputation may be applied to complex missing data patterns. 

Continuous variables like vitals and lab results will be standardized to a uniform scale when possible, to ensure consistent input across models. In cases where they do not follow normal distributions we will try transformations, such as logarithmic transformations, to obtain a more similar one.
In the cases where this does not succeed, we will use transformations to other distributions, such as exponential or beta distributions.
After this, outliers the data will be identified and addressed to avoid skewing model predictions.

Lastly, class imbalance is expected, with fewer severe cases than non-severe, so we will use methods such as SMOTE to balance the training data. 

#### Feature Subset Selection
To improve model performance and ensure interpretability, we will use feature selection techniques to create several subsets of features to train the models. 

Firstly, filter feature selection – specifically Recursive Feature Elimination (RFE) – will be used to remove less important features, using metrics of feature relevance, such as Chi-Square and Mutual Information.

Secondly, we will employ a wrapper method for feature subset selection (FSS) using a small Multi-Layer Perceptron (MLP) as the evaluation model, combined with an annealing-based search strategy. This approach will help explore the feature space more efficiently by allowing for both local and global exploration, minimizing the risk of getting trapped in suboptimal feature combinations.

The annealing process begins by randomly selecting a subset of features and training the MLP model on this subset. The performance is evaluated using cross-validation, and at each iteration, we make small random adjustments to the feature set by adding or removing features. The modified feature subset is then evaluated, and the decision to accept the new subset is based on a probability that decreases over time (the "cooling schedule"), allowing for less optimal solutions early in the process but gradually focusing on the best-performing feature sets as the method progresses.

The MLP will be kept shallow to reduce complexity and avoid overfitting during the search process. With its ability to capture nonlinear feature interactions, the small MLP provides the flexibility needed for finding optimal feature combinations while still maintaining a manageable model size.

This annealing-based approach will allow us to efficiently search the feature space for a globally optimal subset, balancing exploration and exploitation to ensure that we select the most relevant features for predicting severe COVID-19 outcomes. The final subset of features will be used for model training and evaluation in later stages of the project.

#### Modeling
Several machine learning models will be explored during the modeling phase. We will begin with logistic regression as a baseline due to its simplicity and interpretability, allowing us to understand which variables are critical to predicting severe cases. Next, we will implement more advanced models like Random Forest, which combines multiple decision trees for higher accuracy while providing insight into feature importance. For further refinement, Gradient Boosting Machines (such as XGBoost) will be used to improve accuracy by correcting errors iteratively. If necessary, we will explore deep learning models, though their complexity and reduced interpretability mean they will only be considered if they show significant performance gains.

#### Evaluation Metrics
To evaluate model performance, we will use a set of carefully chosen metrics. We will use ROC-AUC, Cohen's Kappa, Sensitivity (or True Positive Rate) and F2 scores to evaluate the models. In cases where there are clear optimal model, we will priorize Sensitivity scores.

Robustness will be assured through the utilization of bias-corrected 10-fold cross-validation to assure these scores accurately represent the behavior of the model in real world cases. 

## 4. Work Plan & Detailed Task Breakdown

- **Task Breakdown**:
  - **Data Description and Understanding**: [Add analysis of the dataset (demographics, clinical data, ICU admissions, etc.) including key features, distributions, and data quality assessment.]
  - **Data Preprocessing**: [Explain steps to prepare the data, including handling missing data, encoding categorical variables, scaling numerical data.]
  - **Exploratory Data Analysis (EDA)**: [Add a summary of EDA to understand relationships, identify patterns, and discover potential predictors of severe illness.]
  - **Model Development and Testing**: [Describe the process of building, training, and validating the predictive model, including model selection and tuning.]
  - **Model Implementation**: [Detail how the model will be integrated into hospital systems for real-time predictions.]
  - **Model Evaluation**: [Provide methods to evaluate model performance, such as accuracy, precision, recall.]
  - **Risk Assessment and Mitigation**: [Identify risks and strategies to mitigate them, e.g., data privacy, model bias.]
  - **Final Reporting and Presentation**: [Prepare a comprehensive report and presentation for stakeholders.]

- **Work Packages**: [Organize tasks into work packages reflecting different phases of the project, from data understanding to model deployment.]

- **Gantt Chart**: [Provide a Gantt chart that visually represents the timeline, highlighting key milestones and deadlines.]

- **Budget**: [Estimate the budget for the project, including costs such as computational resources, human resources.]

## 5. Risk Analysis

- **Data Risks**: [Identify issues with incomplete, missing, or biased data.]
- **Model Risks**: [Detail potential biases in predictions, accuracy issues, scalability.]
- **Ethical/Privacy Risks**: [Address challenges related to sensitive patient data and privacy regulations compliance.]
- **Operational Risks**: [Identify challenges in implementing the model in real hospital environments and training staff.]

## 6. Viability Analysis

- **Technological Feasibility**: [Assess the availability of data, computational resources, and technical skills.]
- **Financial Feasibility**: [Determine if the potential cost justifies the expected benefits.]
- **Operational Feasibility**: [Analyze how easily the model can be implemented in a hospital setting.]

### Final Deliverables

- **Project Plan Document**: [Include sections like introduction, cost/benefit analysis, data science framing, detailed task breakdown, Gantt chart, risk analysis, and viability analysis.]
- **Presentation**: [Prepare a short presentation (5-10 slides) summarizing the project plan.]

### Evaluation Criteria

- **Clarity**: [Ensure the problem and project goals are well explained.]
- **Feasibility**: [Assess the practicality of the project plan, including timeline and budget.]
- **Completeness**: [Ensure all sections are included and properly detailed.]
- **Data Understanding**: [Provide a deep analysis and description of the dataset.]


