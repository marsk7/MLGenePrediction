# Final Choices
___
## Pre-processing methods
* ### Missing values
The missing values are imputed by all values, use the **median** for numerical features, and use the **most frequent value** for nominal features.

* ### Normalization: 
Use **standardization** for numerical features. Use **pipeline** to handle the imputation and normalization.

* ### Outlier Detection: 
use **Local outlier factor (LOF)** 

* ### Outlier handler: 
Use the **Max Min value** to mask the outliers


## Classification models
* **Voting classifier** either includes **Decision Tree classifier** and **Random Forest classifier**. 

After testing the four classifiers, and comparing with their f1-scores, Get the result: <br>
Random Forest classifier and Decision Tree classifier performed better in the single classifier training. kNN and Naive Bayes need to use different preprocessing tech (MinMax scale) to get the similar score, but perform poor on the split test set.
To achieve a better performance and robust results, combine Random Forest and Decision Tree by voting classifier.


## Hyperparameters
* **LOF outlier detection:**: the hyperparameters are tuned by visualization result: `n_neighbors=5, contamination=0.03`.

* **DecisionTreeClassifier:**: the hyperparameters are tuned by GridsearchCV: `max_depth=2, min_samples_leaf=4, min_samples_split=2`.

* **RandomForestClassifier:** the hyperparameters are tuned by GridsearchCV: `n_estimators=100, max_depth=7, min_samples_split=5, min_samples_leaf=2`.


# Environment Description
___
* Operating system: **macOS 12.7.6** <br>

* Programming language: **Python 3.9** 

* Additional installed packages: **Pandas, Numpy, sklearn(scikit-learn), matplotlib, scipy**.


# Reproduction Instructions
___
## Reproduce the result
To reproduce the result, just run `main.py` file through python3, which will generate result file `s4785581.infs4203` under the same directory. (Need to install all the requirement)

## Reproduce hyperparameter tuning
* ### LOF outlier detection hyperparameters:
Uncomment `main.py` line 61: 
```python
# outlier_lof(X_train_processed)
```
and run `main.py` file, which will generate a sort of matplot figures of outliers with different hyperparameters. 

*(It is better to comment the rest of the code.)

* ### Decision Tree Classifier:
Uncomment `main.py` line 120: 
```python
# dt_hyper(X_train_processed, y_train)
```
and run `main.py` file, which will output hyperparameters as:
```
Best parameters for Decision Tree: {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 2}
Best F1 score for Decision Tree: 0.7912934472934472
```
*(It is better to comment the rest of the code.)

* ### Decision Tree Classifier:
Uncomment `main.py` line 121: 
```python
# rf_hyper(X_train_processed, y_train)
```
and run `main.py` file, which will output hyperparameters as:
```
Best parameters for Random Forest: {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
Best F1 score for Random Forest: 0.7911900008435735
```
*(It is better to comment the rest of the code.)

* ### Evaluation
Evaluation codes are in `main.py` **line 143 - line 155**, once run `main.py`, the console will print out the score of the voting classifier as:
```
Average accuracy: 0.9555
Average precicion: 0.9553
Average recall: 0.6645
Average f1_score: 0.7823
```
and the score on the split test set as:
```
F1 score on Test Set: 0.6780
Accuracy on Test Set: 0.9406
```


# Additional Justifications
___
kNN and Naive Bayes classifier are included in the `hyperparameter_tune.py` file, as well as other outlier detection methods have compared.
# MLGenePrediction
