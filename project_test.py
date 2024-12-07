import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from hyperpara_classifier import find_optimal_eps, outlier_visualize, \
    outlier_lof, outlier_dbscan, outlier_gaussian, outlier_NearestNeighbors, outlier_kmeans, outlier_IsolationForest, \
    dt_hyper, rf_hyper, knn_hyper, nb_hyper


# Phase I
# Data Loading and Preprocessing
# Load the data
train_data = pd.read_csv('DM_project_24.csv')

# Separate features and target
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
# X.to_csv('checkpre.csv')

# Separate numerical and categorical features
numerical_features = X.columns[:103]
categorical_features = X.columns[103:105]

# Using Stratified sampling split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Class-specific imputation for numerical features
def class_specific_imputation(X, y, numerical_features):
    X_imputed = X.copy()
    for class_label in y.unique():
        class_mask = (y == class_label)
        X_class = X_imputed.loc[class_mask, numerical_features]
        # Impute missing values with the class-specific mean
        imputer = SimpleImputer(strategy='median')
        X_imputed.loc[class_mask, numerical_features] = imputer.fit_transform(X_class)
    return X_imputed

# Apply class-specific imputation to the numerical features
X_class_imputed = class_specific_imputation(X_train, y_train, numerical_features)
X_test_imputed = class_specific_imputation(X_test, y_test, numerical_features)

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Better for DT, RF
    # ('scaler', MinMaxScaler())    # Better for KNN, GNB
    # ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data and transform both training and testing data
X_train_processed = preprocessor.fit_transform(X_class_imputed)
X_test_processed = preprocessor.transform(X_test_imputed)

# Apply SMOTE to balance the classes through resampling
smote = SMOTE(random_state=42)
X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)

# Apply undersampling to balance the classes
undersampler = RandomUnderSampler(random_state=42)
X_train_processed, y_train = undersampler.fit_resample(X_train_processed, y_train)


# Outlier Detection
# find_optimal_eps(X_train_processed)

# Hyperparameter Tuning for Outlier detection
# outlier_lof(X_train_processed)
# outlier_dbscan(X_train_processed)
# outlier_gaussian(X_train_processed)
# outlier_NearestNeighbors(X_train_processed)
# outlier_kmeans(X_train_processed)
# outlier_IsolationForest(X_train_processed)



'''
# DBSCAN outlier detection
dbscan = DBSCAN(eps=4, min_samples=20)
# Fit and predict labels
y_pred_outlier = dbscan.fit_predict(X_train_processed)
# Separate outliers
outliers = y_pred_outlier == -1
'''



# LOF outlier detection
# Set up the LOF model with parameters such as n_neighbors
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.03)  # Adjust contamination as necessary

# Fit the LOF model to the dataset and predict outliers (-1 for outliers, 1 for inliers)
y_pred_outlier = lof.fit_predict(X_train_processed)

# Extract the outlier scores (negative outlier factor values)
X_scores = -lof.negative_outlier_factor_

# Identify the indices of the outliers
outliers = np.where(y_pred_outlier == -1)[0]

# Display the number of outliers detected
print(f"Number of outliers detected in the training set: {len(outliers)}")


# Visualize the outlier distribution
# outlier_visualize(X_train_processed, y_pred_lof)


# Hyperparameter Tuning for Outlier detection
# outlier_hyperpara_tune(X_train_processed)


# Handle outliers: Remove
# X_no_outliers = X_train_processed[y_pred_outlier != -1]
# y_no_outliers = y_train[y_pred_outlier != -1]
# X_train_processed = X_no_outliers
# y_train = y_no_outliers


# Handle outliers by Max Min cap
def outlier_handler(X_train_processed):
    # Apply the same outlier capping method to the test data
    for i in range(X_train_processed.shape[1]):
        # Use the min and max values from the training data (without outliers)
        max_value = X_train_processed[y_pred_outlier != -1, i].max()
        min_value = X_train_processed[y_pred_outlier != -1, i].min()

        # Cap the values in the test set
        X_train_processed[:, i] = np.where(X_train_processed[:, i] > max_value, max_value, X_train_processed[:, i])
        X_train_processed[:, i] = np.where(X_train_processed[:, i] < min_value, min_value, X_train_processed[:, i])

    return X_train_processed

X_train_processed = outlier_handler(X_train_processed)

# Define score as the scoring metric
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precicion': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score)
}



# dt_hyper(X_train_processed, y_train)
# rf_hyper(X_train_processed, y_train)
# knn_hyper(X_train_processed, y_train)
# nb_hyper(X_train_processed, y_train)




# Use preprocessed data
# X_train_processed, y_train, X_test_processed, y_test = preposess()

# Apply Hyperparameter for models
models = {
    # 'Decision Tree': DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42),
    # 'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=11, min_samples_split=2, min_samples_leaf=2, class_weight='balanced', random_state=42),
    # 'k-Nearest Neighbor': KNeighborsClassifier(n_neighbors=2, p=2),
    # 'Naïve Bayes': GaussianNB(var_smoothing=np.float64(0.016681005372000558))

    # 'Decision Tree': DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, min_samples_split=2, random_state=42),
    # 'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2, min_samples_leaf=4, class_weight='balanced', random_state=42),
    # 'k-Nearest Neighbor': KNeighborsClassifier(n_neighbors=1, p=1),
    # 'Naïve Bayes': GaussianNB(var_smoothing=np.float64(0.046415888336127725))

    # SMOTE & Underresample
    'Decision Tree': DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=11, min_samples_split=2, min_samples_leaf=2, class_weight='balanced', random_state=42),
    'k-Nearest Neighbor': KNeighborsClassifier(n_neighbors=2, p=2),
    'Naïve Bayes': GaussianNB(var_smoothing=np.float64(1e-12))
}




# Set up the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('decision_tree', models['Decision Tree']),
        ('random_forest', models['Random Forest']),
        ('knn', models['k-Nearest Neighbor']),
        ('naive_bayes', models['Naïve Bayes'])
    ],
    weights=[1, 2, 1, 1],
    # voting='soft'  # Use 'hard' for majority voting or 'soft' for probability-based voting
    voting='hard'
)

# Train the voting classifier
voting_clf.fit(X_train_processed, y_train)

# Evaluate the voting classifier with cross-validation to get the F1 score
cv_scores = cross_val_score(voting_clf, X_train_processed, y_train, cv=5, scoring='f1')
print(f"Average F1 score (Voting Classifier): {cv_scores.mean():.4f}")

# Evaluate on the test set
y_pred = voting_clf.predict(X_test_processed)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 score (Test Set): {f1:.4f}")
print(f"Accuracy (Test Set): {accuracy:.4f}")





'''
# Cross-Validation and Model Evaluation
# Iterate over each model and evaluate using cross-validation
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    # Perform cross-validation
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for scorer_name, scorer in scorers.items():
        scores = cross_val_score(model, X_train_processed, y_train, cv=stratified_cv, scoring=scorer)
        # print(f'{scorer_name}: {scores}')
        print(f'Average {scorer_name} for {model_name}: {np.mean(scores):.4f}')
    print('-' * 50)




# Predict the labels for the test set
# best_model = models['Decision Tree'].fit(X_train_processed, y_train)
# best_model = models['Random Forest'].fit(X_train_processed, y_train)
# best_model = models['k-Nearest Neighbor'].fit(X_train_processed, y_train)
# best_model = models['Naïve Bayes'].fit(X_train_processed, y_train)
for model_name, model in models.items():
    best_model = model.fit(X_train_processed, y_train)
y_test_pred = best_model.predict(X_test_processed)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)

# Print the results
print(f"Accuracy: {accuracy:f}")
print(f"F1 Score: {f1:f}")
print(f"Precision: {precision:f}")
print(f"Recall: {recall:f}")
'''

'''

# Phase II
# Load the test data
test_data = pd.read_csv('test_data(1).csv')

# Apply the same preprocessing steps as used for training data
X_test_processed = preprocessor.transform(test_data)

# Fit the model with the training data
# best_model = models['Random Forest'].fit(X_train_processed, y_train)

# Use the best-trained model (e.g., Random Forest) to predict the labels for the test data
y_test_pred = best_model.predict(X_test_processed)

# Evaluate the model using cross-validation on the training data
accuracy = cross_val_score(best_model, X_train_processed, y_train, cv=5, scoring=scorers['accuracy'])
f1 = cross_val_score(best_model, X_train_processed, y_train, cv=5, scoring=scorers['f1_score'])

# Calculate the mean accuracy and F1 score
mean_accuracy = round(accuracy.mean(), 3)
mean_f1 = round(f1.mean(), 3)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(y_test_pred, columns=['Prediction'], dtype=int)
predictions_df.to_csv('test_data_predictions.csv', index=False, header=False, lineterminator=',\n')

print(predictions_df.head())
print(predictions_df.tail())

# Open the file in append mode to add the evaluation row
with open('predictions_test_data.csv', 'a') as f:
    f.write(f"{mean_accuracy},{mean_f1}")


with open('predictions_test_data.csv', 'r') as file:
    lines = file.readlines()
    print(lines[:5])
    print(lines[-5:])

with open('../result_report_example.infs4203', 'r') as file:
    lines = file.readlines()
    print(lines[:5])
    print(lines[-5:])
'''

