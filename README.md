# CalEX - Predicting Amount Of Calories Burnt using ML Models
## Overview

This project explores the application of machine learning algorithms, including Linear Regression, Random Forest Regressor, and XGBoost, for predictive modeling. 
By leveraging these algorithms, we aim to build robust predictive models capable of handling complex datasets and making accurate predictions.
We tested 3 different model alogrithms and XG Boost Regressor gives the best result.

CalEx Jupyter file goes through step by step build of the project along with explanation while final.py is the finalized code after all the testing and evaluating.

## About Algorithms

## *Linear Regression*
- Linear Regression is a simple and widely used statistical technique for predictive modeling.
- It assumes a linear relationship between the independent variables (features) and the dependent variable (target).
- The model learns the coefficients of the linear equation that best fits the observed data, minimizing the sum of squared differences between the actual and predicted values.
- Linear Regression is interpretable and computationally efficient, making it suitable for datasets with many features.

## *Random Forest Regressor*
- Random Forest is an ensemble learning method that builds a collection of decision trees during training.
- Each tree is trained on a random subset of the training data and a random subset of features, leading to diverse trees.
- During prediction, the output of each tree is averaged (regression) to produce the final prediction.
- Random Forest is robust to overfitting, handles non-linear relationships well, and can capture complex interactions in the data.
- It's less prone to bias and variance compared to individual decision trees, making it a popular choice for regression tasks.

## *XG Boost Regressor*
- XGBoost is an optimized implementation of gradient boosting, a powerful ensemble learning technique.
- It builds a collection of weak predictive models (typically decision trees) sequentially, where each new model corrects the errors made by the previous ones.
- XGBoost uses gradient descent optimization techniques to minimize a loss function, improving model performance iteratively.
- It incorporates regularization techniques to prevent overfitting and handles missing values internally.
- XGBoost is known for its efficiency, scalability, and state-of-the-art performance in various machine learning competitions and real-world applications.

## Installation

To run the code in this project, you'll need the following dependencies:
- Python (version >= 3.6)
- NumPy
- Pandas
- Seaborn
- Scikit-learn
- XGBoost

**Clone this repository to your local machine:**

```bash
git clone https://github.com/Astolsko/CalEx-Step-by-step-build-of-a-machine-learning-model-to-predict-calories-burnt.git
```

## Contributing

Contributions to this project are welcome! Feel free to submit bug reports, feature requests, or pull requests through GitHub.
