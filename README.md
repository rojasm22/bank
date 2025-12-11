# Bank: Machine Learning Model Exploration and Optimization

Welcome to the **Bank** project, a comprehensive machine learning exploration implemented in Python using Google Colab. This project applies and evaluates a variety of machine learning models and optimization techniques, while showcasing preprocessing capabilities and insightful feature engineering to address data leakage issues.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies and Libraries](#technologies-and-libraries)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Bank** project explores the performance of multiple machine learning models and their hyperparameter optimization techniques. It delves into advanced preprocessing, feature engineering, and model evaluation to analyze classification tasks while addressing typical challenges, such as discovering and mitigating data leakage.

The key objectives are:
- Model benchmarking: KNN, Trees, SVM, Logistic Regression, Random Forest, and XGBoost.
- Hyperparameter Optimization: Optuna, Randomized Search, and Grid Search.
- Dimensionality Reduction: PCA with `f_classif` and `mutual_info_classif` using `SelectKBest`.
- Feature Engineering: Implementing transformations and identifying data leakage for better model reliability.

## Features

- **Preprocessing**: Advanced handling of missing data, encoding categorical variables, and scaling numerical features.
- **Hyperparameter Optimization (HPO)**: Utilization of cutting-edge tools such as Optuna and Grid/Randomized Search for optimal model tuning.
- **Dimensionality Reduction**: Application of PCA and feature selection techniques using statistical measures.
- **Model Comparison**: In-depth analysis of algorithms like KNN, decision trees, SVM, logistic regression, random forest, and gradient boosting (XGBoost).
- **Visualization**: Comprehensive analytics with confusion matrices, ROC curves, parameter importance, and optimization history plots.
- **Feature Engineering**: Exploration of the dataset to identify data leakage and improve model robustness.

## Technologies and Libraries

The following libraries/tools were used in this project:

### Core Libraries:
- **Python**: Core programming language.
- **Google Colab**: Hosted Jupyter notebook environment for efficient model experimentation.
- **Pandas, NumPy**: Data analysis and manipulation.
- **Matplotlib, Seaborn**: Visualization tools.

### Preprocessing:
- `sklearn.impute` (SimpleImputer, MissingIndicator, IterativeImputer)
- `sklearn.preprocessing` (OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder, FunctionTransformer)
- `sklearn.pipeline` and `sklearn.compose`

### Model Training:
- `sklearn.neighbors` (KNN)
- `sklearn.tree` (Decision Trees)
- `sklearn.linear_model` (Logistic Regression)
- `sklearn.svm` (Support Vector Machines)
- `sklearn.ensemble` (Random Forest, BaggingClassifier)
- **XGBoost** (XGBClassifier)
- `sklearn.dummy` (Dummy Classifiers)

### Model Evaluation:
- Metrics: `accuracy_score`, `classification_report`, `f1_score`, etc.
- Visualization: `ConfusionMatrixDisplay`, `RocCurveDisplay`

### Hyperparameter Optimization:
- **Optuna**: `OptunaSearchCV`, `optuna.visualization`
- `GridSearchCV`, `RandomizedSearchCV`

### Advanced Feature Selection:
- `SelectKBest` using **f_classif** and **mutual_info_classif**

## Requirements

- Python 3.8 or higher.
- Google Colab Environment.

### Package Installation:
Please refer to the `requirements.txt` file for the list of dependencies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rojasm22/bank.git
   ```

2. Upload your dataset to Google Colab:

   ```python
   from google.colab import files
   uploaded_file = files.upload()
   ```

3. Install required dependencies within your Colab notebook:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook in Google Colab.
2. Follow the pipeline for:
   - Data Preprocessing
   - Feature Engineering
   - Training and Evaluating Models
   - Hyperparameter Optimization
   - Dimensionality Reduction with PCA
3. Visualize and analyze the results using built-in plots and metrics.

## Project Structure

```
.
├── data/                    # Example datasets
├── notebooks/               # Google Colab notebooks
│   └── main_project.ipynb   # Main notebook for the project
├── plots/                   # Generated plots (e.g., confusion matrices, ROC curves)
├── scripts/                 # Helper scripts for preprocessing and visualization
└── requirements.txt         # List of dependencies
```

## Contributing

Contributions are welcome! If you have suggestions, improvements, or interesting ideas related to model selection or optimization, feel free to create an issue or open a pull request.

## License

This repository does not currently specify a license. Please add one to clarify usage rights.

---

Dive into this project and discover the art of tuning machine learning models effectively with cutting-edge techniques!
