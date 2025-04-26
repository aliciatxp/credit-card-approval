# Credit Card Approval Prediction
Data source: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

## Overview
This project develops a machine learning model to predict whether an applicant should be approved for a credit card based on their personal and financial information. As credit card delinquencies rise amid economic uncertainty, with consumer debt reaching $17.5 trillion in 2023, banks face significant financial losses due to credit risk. Our solution aims to help financial institutions identify creditworthy applicants more effectively, reducing potential losses while ensuring deserving applicants aren't wrongly rejected.

## Dataset
The dataset contains two main tables:
- **application_record.csv**: Contains applicant's personal information including employment details, ownership statuses, and family status
- **credit_record.csv**: Contains the applicant's credit history, including payment status

The dataset mimics the type of data used by banks for credit scoring and risk assessment tasks, with over 400,000 instances providing a large sample for predictive modeling.

## Problem Statement
We aim to develop a machine learning model that predicts whether an applicant is classified as 'good' or 'bad' for credit card approval based on the features of the dataset. Key questions addressed:
- What makes an applicant 'good' or 'bad'?
- Which features are more important for approval?
- Which supervised machine learning model performs best for this task?
- How can we handle unbalanced data?

## Methodology

### Data Preprocessing
1. **Data Cleaning**
   - Handled missing values (30% in occupation_type)
   - Addressed duplicate IDs
   - Standardized categorical and binary columns
   - Dealt with outliers (e.g., employment_duration = -1001)

2. **Labeling Strategy**
   ![labelling2](https://github.com/user-attachments/assets/1485c38a-8dec-42bc-b6b9-117654c76cd7)

3. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Ordinal encoding for education type
   - Correlation analysis to drop highly correlated features

4. **Handling Imbalanced Data**
   - Used SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)

5. **Dimensionality Reduction**
   - Applied Principal Component Analysis (PCA) to reduce features while preserving important patterns

### Models Implemented
1. **Decision Trees**
   - Optimized hyperparameters: max_depth, min_samples_split, min_samples_leaf
   - Used GridSearchCV for hyperparameter tuning

2. **Random Forests**
   - Optimized hyperparameters: n_estimators, max_features, max_depth
   - Ensemble method to reduce variance and overfitting

3. **Logistic Regression**
   - Optimized hyperparameters: C (inverse of regularization strength), penalty type, solver, max_iterations

### Evaluation Metrics
- Accuracy
- Precision
- Recall (primary focus for minimizing bank losses)
- F1-Score
- ROC-AUC

## Results
Decision Trees outperformed both Random Forests and Logistic Regression, particularly in recall, which is critical for minimizing false negatives (wrongly predicting rejected applicants as accepted).

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | ROC-AUC |
|-------|----------|-------------------|----------------|------------------|---------|
| Decision Tree | 0.97 | 0.61 | 0.63 | 0.62 | 0.76 |
| Random Forests | 0.98 | 0.63 | 0.60 | 0.62 | 0.76 |
| Logistic Regression | 0.62 | 0.50 | 0.51 | 0.40 | 0.49 |

## Future Improvements
1. Apply SMOTE-NC and PCA within each fold during cross-validation to avoid data leakage
2. Refine labeling criteria with more context about the dataset
3. Acquire more data, particularly for the minority class
4. Explore additional models and ensemble techniques

## Dependencies
- Python
- scikit-learn
- pandas
- numpy
- LIME (for model interpretation)
- imbalanced-learn (for SMOTE-NC)

## Contributors
- Alicia
- Bernice
- Chelsea
- Haidah
- Mujie
- Shauna
