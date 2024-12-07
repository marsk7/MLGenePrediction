import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as stats


# Define score as the scoring metric
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precicion': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score)
}

def find_optimal_eps(X_train_processed):
    # Calculate the nearest neighbors for each point
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_train_processed)
    distances, indices = neighbors_fit.kneighbors(X_train_processed)

    # Sort distances to find the "elbow" point
    distances = np.sort(distances[:, 4])
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title("K-Distance Graph to Determine Optimal eps")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("5-Nearest Neighbor Distance")
    plt.show()


# Hyperparameter Tuning for Outlier detection
# LOF Outlier Detection Hyperparameter Tuning
def outlier_lof(X_train_processed):
    for contamination in np.arange(0.01, 0.16, 0.02):
        for n_neighbors in np.arange(3, 12, 2):
            # Set up the LOF model with parameters
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

            # Fit the LOF model to the dataset and predict outliers
            y_pred_lof = lof.fit_predict(X_train_processed)

            # Extract the outlier scores (negative outlier factor values)
            X_scores = -lof.negative_outlier_factor_

            # Set up the figure
            plt.figure(figsize=(10, 6))
            plt.scatter(X_train_processed[:, 0], X_train_processed[:, 1], c=X_scores, s=30, cmap='viridis', alpha=0.7)
            plt.colorbar(label='LOF Score')
            plt.title(f'LOF Anomaly Detection with contamination={contamination:.2f} and k={n_neighbors}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')

            # Mark the detected outliers in red
            plt.scatter(X_train_processed[y_pred_lof == -1, 0],
                        X_train_processed[y_pred_lof == -1, 1],
                        c='red', marker='x', label='Detected Outliers')
            plt.legend()
            plt.show()

    # Best parameters for LOF: {'contamination': 0.03, 'n_neighbors': 5}


# DBSCAN Outlier Detection Hyperparameter Tuning
def outlier_dbscan(X_train_processed):
    # Define the range for eps and min_samples parameters
    eps_range = np.arange(4, 5, 0.5)  # Adjust the range and step size as needed
    min_samples_range = [10, 20, 30]  # Different values for min_samples
    # eps=4
    # min_samples=10, 20

    for eps in eps_range:
        for min_samples in min_samples_range:
            # Initialize DBSCAN with the current parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)

            # Fit and predict labels
            y_pred_dbscan = dbscan.fit_predict(X_train_processed)

            # Separate outliers and inliers
            outliers = y_pred_dbscan == -1
            inliers = y_pred_dbscan != -1

            # Plotting
            plt.figure(figsize=(8, 6))
            plt.scatter(X_train_processed[inliers, 0], X_train_processed[inliers, 1], c='blue', s=20, alpha=0.5, label='Inliers')
            plt.scatter(X_train_processed[outliers, 0], X_train_processed[outliers, 1], c='red', marker='x', s=50, label='Detected Outliers (DBSCAN)')
            plt.title(f"DBSCAN Outlier Detection with eps={eps:.2f} and min_samples={min_samples}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.show()


# Univariate Gaussian Outlier Detection Hyperparameter Tuning
def outlier_gaussian(X):
    threshold_range = [2, 3, 5, 7, 9, 11, 13, 15]
    outliers_count = []
    for threshold in threshold_range:
        likelihoods = np.zeros_like(X, dtype=float)
        filters = np.zeros_like(X, dtype=float)
        for col in range(X.shape[1]):
            mean = np.mean(X[:, col])
            std_dev = np.std(X[:, col])
            likelihoods[:, col] = stats.norm.pdf(X[:, col], loc=mean, scale=std_dev)
            filters[:, col] = stats.norm.pdf(mean + (threshold * std_dev), loc=mean, scale=std_dev)

        condition_mask = np.any(likelihoods < filters, axis=1)
        outliers = X[condition_mask]
        outliers_count.append(np.sum(condition_mask))

        # outlier figure
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
        plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', label='Anomalies')
        plt.title(f"Gaussian Model Anomaly Detection for threshold: {threshold}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    # outlier elbow figure
    plt.figure(figsize=(8, 6))
    plt.plot(threshold_range, outliers_count, marker='o')
    plt.title("Gaussian Elbow Method")
    plt.xlabel("Threshold (Standard Deviation Multiplier)")
    plt.ylabel("Number of Outliers Detected")
    plt.show()
    # threshold=5


# Nearest Neighbors Outlier Detection Hyperparameter Tuning
def outlier_NearestNeighbors(X):
    n_neighbors_range = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 100]
    outliers_count = []
    for n_neighbors in n_neighbors_range:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distance_score = distances[:, n_neighbors - 1]
        threshold = np.percentile(distance_score, 95)
        outliers = distance_score > threshold
        outliers_count.append(np.sum(outliers))

        # outlier figure
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=distance_score, marker='x', cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(label='Distance Score')
        plt.title(f"Distance-based Detection (n_neighbors={n_neighbors})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    # outlier elbow figure
    # plt.figure(figsize=(8, 6))
    # plt.plot(n_neighbors_range, outliers_count, marker='o')
    # plt.title("Distance-based Elbow Method")
    # plt.xlabel("Number of Neighbors (n_neighbors)")
    # plt.ylabel("Number of Outliers Detected")
    # plt.show()
    # A straight line


# K-Means Outlier Detection Hyperparameter Tuning
def outlier_kmeans(X):
    n_clusters_range = [2, 3, 5, 7, 8, 9]
    percentile_range = [80, 85, 90, 95]
    for n_clusters in n_clusters_range:
        for percentile in percentile_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_
            distances = np.min(kmeans.transform(X), axis=1)
            threshold = np.percentile(distances, percentile)
            anomalies = distances > threshold

            plt.figure(figsize=(8, 6))
            plt.scatter(X[:, 0], X[:, 1], c=distances, cmap='viridis', alpha=0.7)
            plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', marker='x', label='Anomalies')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='orange', marker='*', s=100, label='Cluster Centers')
            plt.colorbar(label='Distance to Centroid')
            plt.title(f"K-Means Detection (n_clusters={n_clusters}, percentile={percentile})")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.show()

    # n_clusters=7, percentile=95


# Isolation Forest Outlier Detection Hyperparameter Tuning
def outlier_IsolationForest(X):
    n_estimators_range = [50, 100, 150]
    contamination_range = [0.05, 0.1, 0.13, 0.15]
    for n_estimators in n_estimators_range:
        for contamination in contamination_range:
            iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            y_pred = iso_forest.fit_predict(X)

            plt.figure(figsize=(8, 6))
            plt.scatter(X[:, 0], X[:, 1], c='blue', s=20, alpha=0.7, label='Inliers')
            plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', marker='x', s=50, label='Detected Outliers')
            plt.title(f"Isolation Forest Detection (n_estimators={n_estimators}, contamination={contamination})")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.show()

    # n_estimators=100, contamination=0.1


# Visualize the outlier distribution
def outlier_visualize(X_train_processed, y_pred_lof):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_processed[:, 0], X_train_processed[:, 1], c=y_pred_lof, s=20, cmap='coolwarm', edgecolor="k")
    plt.colorbar(scatter, label='LOF Prediction')

    plt.scatter(X_train_processed[y_pred_lof == -1, 0], X_train_processed[y_pred_lof == -1, 1],
                c='red', marker='x', s=50, label='Detected Outliers')
    plt.scatter(X_train_processed[y_pred_lof == 1, 0], X_train_processed[y_pred_lof == 1, 1],
                c='blue', s=20, label='Inliers', alpha=0.5)

    plt.title("LOF Outlier Detection Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# Hyperparameter Tuning for Decision Tree
def dt_hyper(X, y):
    param_grid_dt = {
        'max_depth': [None, 1, 2, 3, 4, 5],
        'min_samples_split': [2, 3, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # K-Fold
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=stratified_cv, scoring=scorers['f1_score'])
    grid_search_dt.fit(X, y)
    print("Best parameters for Decision Tree:", grid_search_dt.best_params_)
    print("Best F1 score for Decision Tree:", grid_search_dt.best_score_)

    # Best parameters for Decision Tree: {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 2}
    # Best F1 score for Decision Tree: 0.7912934472934472


# Hyperparameter Tuning for Random Forest
def rf_hyper(X, y):
    param_grid_rf = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 7, 9, 10, 12, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 5]
    }
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid_rf, cv=stratified_cv, scoring=scorers['f1_score'])
    grid_search_rf.fit(X, y)
    print("Best parameters for Random Forest:", grid_search_rf.best_params_)
    print("Best F1 score for Random Forest:", grid_search_rf.best_score_)

    # Best parameters for Random Forest: {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
    # Best F1 score for Random Forest: 0.7911900008435735



# Hyperparameter Tuning for kNN
def knn_hyper(X, y):
    pca = PCA(n_components=30)  # Reduce to 30 principal components
    X_reduced = pca.fit_transform(X)

    param_grid_knn = {'n_neighbors': [int(x) for x in np.arange(1, 22, 1)], 'p': [1, 2]}
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=stratified_cv, scoring=scorers['f1_score'])
    grid_search_knn.fit(X_reduced, y)
    print("Best parameters for kNN:", grid_search_knn.best_params_)
    print("Best F1 score for kNN:", grid_search_knn.best_score_)

    # Best parameters for kNN: {'n_neighbors': 8, 'p': 1}
    # Best F1 score for kNN: 0.7912934472934472


# Hyperparameter Tuning for Naïve Bayes
def nb_hyper(X, y):
    param_grid_gnb = {'var_smoothing': np.logspace(-12, -1, 100)}
    gnb = GaussianNB()

    # K-Fold
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid_gnb, cv=stratified_cv, scoring=scorers['f1_score'])
    grid_search_gnb.fit(X, y)
    print("Best parameters for GaussianNB:", grid_search_gnb.best_params_)
    print("Best F1 score for GaussianNB:", grid_search_gnb.best_score_)

    # Best parameters for GaussianNB: {'var_smoothing': np.float64(0.00046415888336127724)}
    # Best F1 score for GaussianNB: 0.7734885509227614

