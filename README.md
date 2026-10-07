# 🩺 Diabetes Prediction — End-to-End Machine Learning

An end-to-end machine learning project that predicts diabetes outcomes using the **Pima Indians Diabetes Dataset**. The project covers data cleaning, exploratory data analysis (EDA), preprocessing, model comparison, hyperparameter tuning, threshold optimization, and model interpretation.

The goal is not just to maximize accuracy, but to understand how model selection and decision thresholds affect the identification of patients who may have diabetes.

## 🎯 Project Objectives

- Perform data understanding and exploratory data analysis.
- Identify and handle invalid zero values in medical features.
- Build a leakage-safe preprocessing pipeline.
- Compare multiple classification algorithms using stratified cross-validation.
- Tune promising models using `GridSearchCV`.
- Evaluate the final model on an untouched test set.
- Optimize the decision threshold to improve recall.
- Interpret model predictions using permutation importance.
- Save the trained model for future predictions.

## 📊 Dataset

**Dataset:** Pima Indians Diabetes Database

The dataset contains 768 observations and eight predictor features. The target variable, `Outcome`, indicates whether diabetes was recorded.

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Two-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Family-history-related diabetes risk score |
| `Age` | Age in years |
| `Outcome` | Target: 1 = diabetic, 0 = non-diabetic |

The dataset represents a specific population of female patients of Pima Indian heritage. Its findings should not automatically be generalized to other populations.

## 🔍 Project Workflow

### 1. Data Understanding and Cleaning

- Inspected dataset dimensions, data types, and descriptive statistics.
- Checked for duplicate records.
- Identified zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` that may represent missing measurements.
- Replaced these invalid zero values with `NaN` for appropriate imputation.

### 2. Exploratory Data Analysis

EDA was performed to investigate:

- Target class distribution.
- Feature distributions and outliers.
- Relationships between features and the target.
- Correlations between clinical measurements.
- Diabetes rates across glucose-level bands.

**Key observations from the notebook:**

- Glucose showed the strongest correlation with the target (approximately 0.49).
- BMI, insulin, and age also showed positive associations with the target.
- Diabetes rates increased across the glucose bands examined.
- Insulin and skin thickness contained substantial missing-value proportions after invalid zeros were identified.

These are exploratory associations, not evidence of causation.

### 3. Preprocessing

The workflow uses an 80/20 stratified train-test split with `random_state=42`.

A Scikit-learn pipeline combines:

- Median imputation using `SimpleImputer`.
- Standardization using `StandardScaler`.
- The selected classification model.

Splitting the data before fitting preprocessing steps helps prevent information from the test set leaking into model training.

### 4. Model Comparison

Seven classification algorithms were evaluated using five-fold stratified cross-validation on the training set.

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable linear classification baseline |
| K-Nearest Neighbors | Distance-based classification |
| Support Vector Machine | Margin-based classification |
| Gaussian Naive Bayes | Probabilistic classification |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting ensemble |

Models were compared using accuracy, recall, F1-score, and ROC-AUC. ROC-AUC was used as the principal model-selection metric.

### 5. Hyperparameter Tuning

`GridSearchCV` with five-fold cross-validation was used to tune selected candidates, including Logistic Regression, SVM, Random Forest, and Gradient Boosting.

The tuned models were compared using cross-validated ROC-AUC. The SVM achieved the highest reported cross-validation score among the selected tuned models and was chosen for final evaluation.

### 6. Final Evaluation

The selected SVM model was evaluated on the held-out test set.

| Metric | Reported result |
|---|---:|
| Test ROC-AUC | **0.805** |
| Recall at 0.50 threshold | Approximately 46% |
| Recall at 0.25 threshold | Approximately 83% |
| Precision at 0.25 threshold | Approximately 55% |
| Accuracy at 0.25 threshold | Approximately 70.1% |

The reported results are rounded values from the notebook and may vary if the data or workflow changes.

**Interpretation:** The default 0.50 decision threshold missed more than half of the diabetic observations in the test set. Lowering the threshold to 0.25 increased recall, at the cost of lower precision and slightly lower accuracy.

This illustrates why accuracy alone is insufficient for evaluating a medical classification model.

### 7. Model Interpretation

Permutation importance was used to estimate the contribution of individual features to predictive performance.

The notebook identified:

- **Glucose** as the most influential feature.
- Pregnancies and BMI as additional useful predictors.
- Smaller contributions from several other features, some of which contain substantial missing data.

Permutation importance measures changes in model performance when a feature is shuffled; it does not establish causal importance.

### 8. Threshold Optimization

Instead of relying exclusively on the default probability threshold, the project uses out-of-fold predictions on the training data to select a threshold that achieves at least 80% recall.

The selected threshold is then applied to the held-out test probabilities.

This makes the decision threshold responsive to the intended screening objective while avoiding threshold selection directly on the test labels.

### 9. Model Persistence

The trained pipeline and selected threshold are saved together using Joblib:

```python
joblib.dump(
    {
        "model": best_model,
        "threshold": THRESHOLD
    },
    "diabetes_model.joblib"
)
```

The notebook also demonstrates a prediction function that accepts patient feature values and returns the estimated probability and threshold-based classification.

## 🛠️ Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Environment:** Jupyter Notebook

## 📁 Project Structure

```text
Diabetes-Prediction/
│
├── dataset/
│   └── diabetes.csv
│
├── notebook/
│   └── Diabetes_Prediction_End_to_End.ipynb
│
├── diabetes_model.joblib
├── README.md
├── requirements.txt
└── .gitignore
```

Adjust the file names and folders to match the actual project. Include the saved model file only if you intend to distribute the generated artifact.

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the Pima dataset CSV in the expected location and name it `diabetes.csv`. The notebook also contains a fallback to a public copy of the dataset if that local file is unavailable.

### 3. Run the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open `Diabetes_Prediction_End_to_End.ipynb` and execute the cells in order.

## 📦 Requirements

Create a `requirements.txt` containing:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

## ⚠️ Limitations

- The dataset contains only 768 observations.
- It represents a specific population and may not generalize to other populations.
- Several clinical features have substantial missingness.
- The test set contains only 154 observations, making performance estimates uncertain.
- The threshold was selected to prioritize recall, which increases false positives.
- The model has not undergone clinical validation or external validation.

This project is intended for educational and portfolio purposes only. It is **not a medical diagnostic tool** and must not be used to make clinical decisions.

## 🔮 Future Improvements

- Extend the project to a larger diabetes dataset with additional demographic and clinical variables.
- Compare performance across datasets using consistent evaluation protocols.
- Explore probability calibration and additional threshold-selection methods.
- Add SHAP-based explanations.
- Build an interactive Streamlit application.
- Evaluate generalization on independent data.

## 📚 Key Learning Outcomes

- Handling invalid values in medical datasets.
- Performing exploratory data analysis.
- Building preprocessing pipelines without data leakage.
- Comparing and tuning classification models.
- Evaluating imbalanced classification problems.
- Understanding precision-recall trade-offs.
- Optimizing decision thresholds.
- Interpreting models with permutation importance.
- Saving trained models for reuse.

## 👨‍💻 Author

**Atharv Singh**

B.Tech in Mechanical Engineering | AIML Minor  
Foundations in Data Science — IIT Madras

Interested in Data Analytics, Machine Learning, and Data Science.

---

**Disclaimer:** This project is for educational purposes only. Predictions should not be interpreted as medical advice or a diagnosis.
