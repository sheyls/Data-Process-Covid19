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

Our objective is to forecast severe COVID-19 illnesses among diabetes patients aged 40 to 6 in a�reproducible�and�interpretable�manner. 
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

In this section, we will delve into:

1.  The actual data we expect to get from each source
2.  the integrability
3.  ethical concerns
4.  completeness
5.  data cleanliness and required reconciliation

#### Data provided

As described in the objective section, our data will be sourced from:

1.  Demographic data.
2.  Clinical data.
3.  Hospital admission data.
4.  Social and Behavioral Determinants of Health (SBDH).

#### Data access and formats

To ensure a smooth and efficient integration of all datasets into our data science pipeline, we have outlined the following expectations and requirements for how the data should be provided to our team. These guidelines will help us manage data integration, access, and formatting effectively.

##### Acceptable formats

We expect data to be delivered in standard formats compatible with our systems to ease their processing and integration. Preferred formats include:

-   CSV: Comma-separated values for structured data, suitable for demographic, clinical, hospital admission, and SBDH data. The CSV files should contain clear column headers and consistent data types.
-   JSON/XML: For semi-structured data, such as social and behavioral determinants of health (SBDH), or if the data contains hierarchical or nested information (e.g., survey results, API outputs).
-   Parquet: For larger datasets that need efficient storage and compression, especially for clinical or hospital admission data.
-   SQL/Database Exports: If data is stored in a relational database, we will accept exports in SQL format or direct access to database dumps (e.g., MySQL, PostgreSQL) where applicable.

##### Metadata

All data files should be well-documented, with data dictionaries provided to explain the meaning of each variable and with time stamping to accurately track when the data was captured.

Every dataset should come with metadata that explains:

-   Field descriptions (e.g., what each column represents).
-   Units of measure, data types, and constraints.
-   Assumptions made during data collection.
-   Data Dictionary: A data dictionary outlining all variables, their types, potential categories, and ranges, if applicable.
-   Codebook: If data contains coded values (e.g., for demographic fields), a codebook explaining the meaning of each code should be included.
-   Source Documentation: Clear documentation of where the data originates (e.g., hospitals, clinical trials, surveys) and how it has been collected.
-   Transformation Records: A log of any transformations (e.g., filtering, aggregating, normalizing) applied to the data from collection to delivery. This is crucial for understanding how raw data has evolved to its final form.
-   Version Control: Assurances that data updates or changes (e.g., new entries, modifications) are tracked with proper version control, so historical data can be reviewed or reverted.

##### Data Access

To facilitate efficient data integration, the data should be accessible by our data science team in a controlled and secure environment. We expect:

-   Single Point of Access: All data should be consolidated in a single location or repository to minimize delays. The preferred options are cloud-based folders or a shared database.
-   Role-Based Access: Access to data should be restricted based on roles, ensuring that only authorized personnel can retrieve sensitive information (e.g., clinical data).
-   Version Control: If the data will be updated periodically (e.g., clinical or hospital admission data), we require a versioning system to track changes and prevent data corruption or inconsistency during analysis.

#### Ethical concerns

Due to the sensitive nature of the data sources involved in this project, we must address the main ethical concerns of the data we will receive: data privacy and confidentiality. All data should have gone through a process of anonymization to ensure that patients cannot be identified. Similarly, only data that is strictly necessary for the analysis will be provided to decrease the probability of identifying individuals through their unique situations. This will be detailed further in the Risk Analysis section.

#### Completeness

Some comments as to how much missing data we will allow. In columnar data, where we can ascertain with metrics the missing data, something like 80% percent of data must be there. In non-columnar data, no idea.

#### Data Quality

**Identifier consistency.**

-   Unique and Stable Identifiers: Each entity (e.g., patient, hospital) should have a unique and persistent identifier (e.g., patient ID). These identifiers should remain consistent across different datasets and data handoffs.
-   No Duplicate Identifiers: The data attributes should not have overlapping identifiers (e.g. two 'Temperature' attributes for one patient).
-   Format Consistency: Identifiers must have consistent formats across datasets. For instance, if an identifier is numeric in one dataset, it should not be alphanumeric in another.

**Unit consistency**. All quantitative measurements (e.g., weight, height, temperature, time, distance) must use consistent units across datasets and should come with clearly stated units for each variable to ensure no misinterpretation of the data during analysis.

**Erroneous inputs.** The data provided should have gone through a basic value validation, checking that all values are within the allowed ranges.

**Missing data.**A policy on the handling of missing data should be provided in order to ensure that all missing values are treated correctly during the data science process.

#### Representative data

To ensure that the model's results will be replicable in practice, we will require that the data used be representative of the predicted target population.

Firstly, each individual sampled should belong to the demographic subset we aim to predict, i.e. patients diagnosed with diabetes aged 40 to 60, and the empirical distribution, both the marginal distribution of the features and the joint distributions, used should closely follow the theoretical distribution of these features in our target population. This sampling should be extensive enough to capture possibly critical subgroups and decrease the margin of error due to the sampling process. This includes having a balanced class distribution.

Secondly, the data should be sampled from time periods and geographical locations relevant to the target population. In our case, we expect the individuals sampled to represent the geographical density distributions of Spain's territory and to be from the last year to correctly represent the rapidly changing nature of COVID-19.

Thirdly, the data should be independent, without duplicates, and avoid highly correlated cases from sampling within family units.

### Approach

In this project, we will consider a range of machine-learning techniques tailored to predict severe COVID-19 outcomes among diabetes patients. The primary focus will be on balancing predictive power with interpretability to ensure the model's results can be understood and trusted by healthcare professionals. Below is an outline of the techniques and methods that will be used.

#### Data Pre-processing

In this phase, we will ensure the quality and consistency of the data before modeling. First, missing data will be handled using appropriate imputation techniques, such as mean or mode imputation for basic patterns, while more sophisticated methods like K-Nearest Neighbors (KNN) imputation may be applied to complex missing data patterns.

Continuous variables like vitals and lab results will be standardized to a uniform scale when possible, to ensure consistent input across models. In cases where they do not follow normal distributions, we will try transformations, such as logarithmic transformations, to obtain a more similar one. In the cases where this does not perform well, we will use transformations to other distributions, such as exponential or beta distributions. After this, outliers in the data will be identified and addressed to avoid skewing model predictions.

Lastly, class imbalance is expected, with fewer severe cases than non-severe, so we will use methods such as SMOTE to balance the training data.

#### Feature Subset Selection

To improve model performance and ensure interpretability, we will use feature selection techniques to create several subsets of features to train the models.

Firstly, filter feature selection specifically Recursive Feature Elimination (RFE) will be used to remove less important features, using metrics of feature relevance, such as Chi-Square and Mutual Information.

Secondly, we will employ a wrapper method for feature subset selection (FSS) using a small Multi-Layer Perceptron (MLP) as the evaluation model, combined with an annealing-based search strategy. This approach will help explore the feature space more efficiently by allowing for both local and global exploration, minimizing the risk of getting trapped in suboptimal feature combinations.

The annealing process begins by randomly selecting a subset of features and training the MLP model on this subset. The performance is evaluated using cross-validation, and at each iteration, we make small random adjustments to the feature set by adding or removing features. The modified feature subset is then evaluated, and the decision to accept the new subset is based on a probability that decreases over time (the "cooling schedule"), allowing for less optimal solutions early in the process but gradually focusing on the best-performing feature sets as the method progresses.

The MLP will be kept shallow to reduce complexity and avoid overfitting during the search process. Its ability to capture nonlinear feature interactions provides the flexibility needed to find optimal feature combinations while still maintaining a manageable model size.

This annealing-based approach will allow us to efficiently search the feature space for a globally optimal subset, balancing exploration and exploitation to ensure we select the most relevant features for predicting severe COVID-19 outcomes. The final subset of features will be used for model training and evaluation in later stages of the project.

#### Modeling

Several machine-learning models will be explored during the modeling phase. We will begin with logistic regression as a baseline due to its simplicity and interpretability, allowing us to understand which variables are critical to predicting severe cases. Next, we will implement more advanced models like Random Forest, which combines multiple decision trees for higher accuracy while providing insight into feature importance. For further refinement, Gradient Boosting Machines (such as XGBoost) will be used to improve accuracy by correcting errors iteratively. If necessary, we will explore deep learning models, though their complexity and reduced interpretability mean they will only be considered if they show significant performance gains.

#### Evaluation Metrics

To evaluate model performance, we will use a set of carefully chosen metrics. We will use ROC-AUC, Cohen's Kappa, Sensitivity (or True Positive Rate), and F2 scores to evaluate the models. Where there are clear optimal models, we will prioritize Sensitivity scores.

Robustness will be assured through the utilization of bias-corrected 10-fold cross-validation to ensure these scores accurately represent the behavior of the model in real-world cases.
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


