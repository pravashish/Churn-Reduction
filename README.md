# Customer Churn Prediction using Logistic Regression (with SMOTE and Weight Balancing)

## Project Overview

This project focuses on predicting customer churn for a telecommunications company using logistic regression. By analyzing historical customer data, including demographics, service usage, and billing information, we aim to identify key factors contributing to churn and develop a predictive model to proactively retain customers. We explore and compare the impact of class imbalance handling techniques: weight balancing and SMOTE.

## Files

* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for this analysis.
* `Customer_Churn_Logistic_Regression.ipynb`: Jupyter Notebook containing the code for data preprocessing, EDA, model training, and evaluation, including implementations of regular logistic regression, weight-balanced logistic regression, and SMOTE-enhanced logistic regression.
* `README.md`: This file, providing an overview of the project and its findings.

## Libraries Used

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical computations.
* `seaborn`: For data visualization.
* `matplotlib`: For creating plots and charts.
* `scikit-learn`: For machine learning tasks, including model training, evaluation, and scaling.
* `imblearn`: For implementing SMOTE (Synthetic Minority Over-sampling Technique).

## Data Preprocessing

* Handled missing values in the 'TotalCharges' column by removing rows with missing data.
* Encoded categorical variables using label encoding.
* Scaled numerical features using StandardScaler.
* Addressed class imbalance using:
    * Logistic Regression with weight balancing.
    * Logistic Regression with SMOTE.

## Exploratory Data Analysis (EDA)

* Explored the distribution of the target variable 'Churn', revealing an imbalanced dataset.
* Investigated the relationship between various features and churn, such as 'Contract', 'MonthlyCharges', and 'tenure'.
* Visualized the data using count plots and box plots to identify patterns and trends:
    * Churn is highest among month-to-month contract users.
    * Customers with higher monthly charges are more likely to leave.
    * Short-tenure customers have a greater churn risk.

## Model Training and Evaluation

* Split the data into training and testing sets.
* Trained and compared three logistic regression models:
    * Regular Logistic Regression.
    * Logistic Regression with weight balancing.
    * Logistic Regression with SMOTE.
* Evaluated the models' performance using:
    * Classification reports (precision, recall, F1-score).
    * Confusion matrices.
    * ROC-AUC scores.
    * Log Loss.

### Classification Report Comparisons

* **Regular Logistic Regression:**
    * Accuracy: 80%
    * Biased towards the majority class (no churn).
    * Good precision for non-churn, but lower recall for churn.
* **Logistic Regression with Weight Balancing:**
    * Accuracy: 74%
    * Significantly improved recall for churn.
    * Lower precision for churn, indicating more false positives.
    * More sensitive to the minority class.
* **Logistic Regression with SMOTE:**
    * Accuracy : 79%
    * Improved recall for churn compared to regular logistic regression.
    * Better balance between precision and recall for churn compared to weight balancing.
    * Provides a good balance between the two classes.

## Key Findings

* **Class Imbalance:** The dataset exhibits a significant class imbalance, impacting model performance.
* **Balancing Techniques:** Weight balancing and SMOTE effectively address class imbalance, improving the model's ability to predict churn.
* **SMOTE Performance:** SMOTE provides a strong and balanced performance, offering a good compromise between precision and recall.
* **Key Churn Factors:** Contract type, monthly charges, and tenure are significant factors influencing churn.

## Conclusion

The logistic regression model, particularly when enhanced with SMOTE, effectively predicts customer churn. The analysis highlights the importance of addressing class imbalance and the significant influence of contract type, monthly charges, and tenure on churn. Targeted retention strategies should consider these factors.

## Future Improvements

* Explore hyperparameter tuning for further model optimization.
* Investigate other advanced resampling techniques.
* Consider other machine learning algorithms.
* Implement a customer retention strategy based on the model's predictions.
* Explore feature engineering to create more predictive features.
