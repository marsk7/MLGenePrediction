# **Genetic Function Classifier Training Based on Machine Learning**  
A project leveraging machine learning techniques for accurate genetic function classification.

---

## **Pre-Processing Methods**  
### 1. **Handling Missing Values**  
- Numerical features: Imputed using the **median**.  
- Nominal features: Imputed using the **most frequent value**.

### 2. **Normalization**  
- Applied **standardization** for numerical features.  
- Utilized a **pipeline** to integrate both imputation and normalization steps.

### 3. **Outlier Detection**  
- Method: **Local Outlier Factor (LOF)**.  

### 4. **Outlier Handling**  
- Approach: Masked outliers using the **Max-Min value** technique.

---

## **Classification Models**  
### Voting Classifier  
- Combines **Decision Tree Classifier** and **Random Forest Classifier** for improved performance.  

### Model Comparison  
After testing four classifiers and evaluating their F1 scores:  
- **Random Forest** and **Decision Tree** outperformed other models during single-classifier training.  
- **k-Nearest Neighbors (kNN)** and **Naive Bayes** required different preprocessing (e.g., MinMax scaling) to achieve similar scores but underperformed on the split test set.  
- **Voting Classifier** demonstrated better robustness and accuracy by combining Decision Tree and Random Forest models.

---

## **Hyperparameters**  
### **LOF Outlier Detection**  
- Tuned parameters:  
  - `n_neighbors=5`, `contamination=0.03`.  
  - Based on visualization results.

### **Decision Tree Classifier**  
- Tuned with GridSearchCV:  
  - `max_depth=2`, `min_samples_leaf=4`, `min_samples_split=2`.

### **Random Forest Classifier**  
- Tuned with GridSearchCV:  
  - `n_estimators=100`, `max_depth=7`, `min_samples_split=5`, `min_samples_leaf=2`.

---

## **Environment Description**  
- **Operating System**: macOS 12.7.6  
- **Programming Language**: Python 3.9  
- **Required Packages**:  
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`  

---

## **Reproduction Instructions**  
### 1. **Reproduce the Results**  
- Run the `main.py` file:  
  ```bash
  python3 main.py
  ```
- The results will be saved as `s12345.infs4203` in the current directory.  
- Ensure all required packages are installed beforehand.

### 2. **Hyperparameter Tuning**  
#### **LOF Outlier Detection**  
- Uncomment line 61 in `main.py`:  
  ```python
  # outlier_lof(X_train_processed)
  ```
- Run `main.py` to generate matplotlib visualizations of outliers for different hyperparameters.  
*(Comment out other code to focus on this step.)*

#### **Decision Tree Classifier**  
- Uncomment line 120 in `main.py`:  
  ```python
  # dt_hyper(X_train_processed, y_train)
  ```
- Run `main.py`. The console output will display:  
  ```
  Best parameters for Decision Tree: {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 2}
  Best F1 score for Decision Tree: 0.7913
  ```

#### **Random Forest Classifier**  
- Uncomment line 121 in `main.py`:  
  ```python
  # rf_hyper(X_train_processed, y_train)
  ```
- Run `main.py`. The console output will display:  
  ```
  Best parameters for Random Forest: {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
  Best F1 score for Random Forest: 0.7912
  ```

### 3. **Evaluation**  
- Evaluation code is located in lines 143–155 of `main.py`.  
- Run `main.py` to display the following results:  
  ```
  Average accuracy: 0.9555
  Average precision: 0.9553
  Average recall: 0.6645
  Average F1 score: 0.7823
  ```
- Split test set results:  
  ```
  F1 score on Test Set: 0.6780
  Accuracy on Test Set: 0.9406
  ```

---

## **Additional Justifications**  
- **kNN** and **Naive Bayes** classifiers, along with alternative outlier detection methods, are included in the `hyperparameter_tune.py` file for comparison purposes.

---

## **Project Highlights**  
This project demonstrates the power of machine learning in classifying genetic functions, with a focus on preprocessing, model optimization, and robust evaluation techniques.

---
