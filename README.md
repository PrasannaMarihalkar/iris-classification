# Iris Classification Project

This project builds a basic machine learning pipeline to classify Iris flowers based on their physical dimensions using the Iris dataset. It serves as a foundational implementation of Supervised Learning.

## Goal
To load data, split it into training and testing sets with strict structural integrity, apply feature scaling, train a classifier, and mathematically validate the model's performance.

## Algorithm Used
* K-Nearest Neighbors (KNN) with `K=5`
* Data normalized using `StandardScaler` to prevent feature dominance.

## Evaluation & Results
The model was validated using an 80/20 train-test split, ensuring the algorithm was tested on completely unseen data. 

* **Accuracy:** 100%
* **F1 Score:** 1.00 across all classes (Setosa, Versicolor, Virginica)
* **Confusion Matrix:** 0 False Positives, 0 False Negatives

## Tech Stack
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib` & `seaborn` (for matrix visualization)
