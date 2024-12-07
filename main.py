import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from hyperparameter_tune import find_optimal_eps, outlier_visualize, \
    outlier_lof, outlier_dbscan, outlier_gaussian, outlier_NearestNeighbors, outlier_kmeans, outlier_IsolationForest, \
    dt_hyper, rf_hyper, knn_hyper, nb_hyper


# Phase I: Data Loading and Preprocessing
# Load the data
train_data = pd.read_csv('DM_project_24.csv')

# Separate features and target
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Separate numerical and categorical features
numerical_features = X.columns[:103]
categorical_features = X.columns[103:105]

# Using Stratified sampling split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Better for DT, RF
    # ('scaler', MinMaxScaler())    # Better for KNN, GNB
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
# X_train_processed = preprocessor.fit_transform(X_class_imputed)
# X_test_processed = preprocessor.transform(X_test_imputed)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Outlier Detection
# Hyperparameter Tuning for Outlier detection
# outlier_lof(X_train_processed)

# find_optimal_eps(X_train_processed)
# outlier_dbscan(X_train_processed)
# outlier_gaussian(X_train_processed)
# outlier_NearestNeighbors(X_train_processed)
# outlier_kmeans(X_train_processed)
# outlier_IsolationForest(X_train_processed)

# LOF outlier detection
def lof_outlier(X_train):
    # Set up the LOF model with parameters such as n_neighbors
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.03)

    # Fit the LOF model to the dataset and predict outliers (-1 for outliers, 1 for inliers)
    y_pred_outlier = lof.fit_predict(X_train)

    # Identify the indices of the outliers
    outliers = np.where(y_pred_outlier == -1)[0]

    # Display the number of outliers detected
    print(f"Number of outliers detected in the training set: {len(outliers)}")

    return y_pred_outlier, outliers


# Call LOF outlier detection
y_pred_outlier, outliers = lof_outlier(X_train_processed)

# Visualize the outlier distribution
# outlier_visualize(X_train_processed, y_pred_outlier)


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

# Call Handle outliers
X_train_processed = outlier_handler(X_train_processed)

# Define score as the scoring metric
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score)
}

# Hyperparameter Tuning for Classifier
# dt_hyper(X_train_processed, y_train)
# rf_hyper(X_train_processed, y_train)
# knn_hyper(X_train_processed, y_train)
# nb_hyper(X_train_processed, y_train)

# Apply Hyperparameter for models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, min_samples_split=2, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42)
}

# Set up the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('decision_tree', models['Decision Tree']),
        ('random_forest', models['Random Forest'])
    ],
    voting='soft'
)

# Train the voting classifier
voting_clf.fit(X_train_processed, y_train)

# Evaluate the voting classifier with cross-validation
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for scorer_name, scorer in scorers.items():
    scores = cross_val_score(voting_clf, X_train_processed, y_train, cv=stratified_cv, scoring=scorer)
    print(f'Average {scorer_name}: {np.mean(scores):.4f}')

# Evaluate on the test set
y_pred = voting_clf.predict(X_test_processed)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 score on Test Set: {f1:.4f}")
print(f"Accuracy on Test Set: {accuracy:.4f}")


# Phase II
# Load the test data
test_data = pd.read_csv('test_data.csv')

# Apply the same preprocessing steps as used for training data
test_processed = preprocessor.transform(test_data)

# Use the trained model to predict the labels for the test data
y_test_pred = voting_clf.predict(test_processed)

# Evaluate the model using cross-validation on the training data
accuracy = cross_val_score(voting_clf, X_train_processed, y_train, cv=stratified_cv, scoring=scorers['accuracy'])
f1 = cross_val_score(voting_clf, X_train_processed, y_train, cv=stratified_cv, scoring=scorers['f1_score'])

# Calculate the mean accuracy and F1 score
mean_accuracy = round(accuracy.mean(), 3)
mean_f1 = round(f1.mean(), 3)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(y_test_pred, columns=['Prediction'], dtype=int)
predictions_df.to_csv('s4785581.infs4203', index=False, header=False, lineterminator=',\n')

# Open the file in append mode to add the evaluation row
with open('s12345.infs4203', 'a') as f:
    f.write(f"{mean_accuracy},{mean_f1}")
