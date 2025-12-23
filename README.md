# Stochastic Gradient Descent from Scratch

## Overview
This project demonstrates the implementation of **Stochastic Gradient Descent (SGD) from scratch** for linear regression using NumPy. The custom implementation is trained on the **Diabetes dataset** from `scikit-learn` and its results are compared with both:

- `LinearRegression` (closed-form solution)
- `SGDRegressor` from `scikit-learn`

The goal is to understand how SGD works internally and verify its correctness by comparing results.

---

## Dataset
- **Dataset**: Diabetes dataset (`sklearn.datasets.load_diabetes`)
- **Features**: 10 numerical features
- **Target**: Disease progression measure
- **Train/Test Split**: 80% / 20%

---

## Implementation Details

### 1. Linear Regression (Baseline)
- Uses `LinearRegression` from `scikit-learn`
- Provides optimal coefficients using the normal equation
- Used as a reference model

### 2. SGD from Scratch
- Custom `SGDRegressor` class implemented using NumPy
- Updates weights using **one random data point at a time**
- Uses Mean Squared Error (MSE) loss
- Parameters:
  - Learning rate
  - Number of epochs
- Outputs learned coefficients and intercept

### 3. SGDRegressor (scikit-learn)
- Uses `SGDRegressor` from `scikit-learn`
- Features are scaled using `StandardScaler`
- Includes optimized learning rate schedules and convergence handling

---

## Comparison
- The custom SGD implementation produces **similar coefficients and intercepts** compared to `SGDRegressor`
- Minor differences occur due to:
  - Random sampling
  - Learning rate choices
  - Feature scaling
- Confirms the correctness of the from-scratch implementation

---

## Key Learnings
- SGD updates parameters incrementally using random samples
- Feature scaling is crucial for stable convergence
- Library implementations are faster and more robust, but manual implementation helps build strong intuition

---

## Technologies Used
- Python
- NumPy
- scikit-learn

---

## Conclusion
This project successfully demonstrates how stochastic gradient descent works internally and validates the custom implementation by comparing it with `scikit-learn`’s optimized `SGDRegressor`.
