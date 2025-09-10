# Linear Regression from Scratch 

A **linear regression model implemented from scratch** in Python. Learn core ML concepts like **gradient descent**, **cost minimization**, and **parameter optimization** without using scikit-learn.

---

## Features 

* Gradient descent optimization
* Custom learning rate
* Feature normalization
* Cost tracking over iterations
* Predict continuous target values

---

## Quick Start 

### Install dependencies

```bash
pip install numpy pandas matplotlib
```

### Load data

```python
import pandas as pd
data = pd.read_csv("house-prices.csv")
x = data["area"]
y = data["price"]
```

### Normalize features

```python
x_scaled = (x - x.mean()) / x.std()
```

### Train model

```python
w, b = 0, 0
alpha = 0.000048
iterations = 20000
m = len(x_scaled)

for i in range(iterations):
    dj_dw = (1/m) * ((w*x_scaled + b - y) * x_scaled).sum()
    dj_db = (1/m) * (w*x_scaled + b - y).sum()
    w -= alpha * dj_dw
    b -= alpha * dj_db
```

### Make predictions & plot

```python
import matplotlib.pyplot as plt
y_pred = w * x_scaled + b

plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()
```

---

## Results 

* Cost decreases steadily over iterations
* Model learns optimal weights `w` and bias `b`
* Can predict new values accurately

---

## Future Improvements 

* Multiple linear regression with more features
* Polynomial regression for non-linear data
* Learning rate scheduling for faster convergence
* Regularization to reduce overfitting

---

## License 📝

MIT License

---
