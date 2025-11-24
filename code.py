import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=load_diabetes()
X=data.data
y=data.target.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train[1])

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

ones_X_train_scaled=np.ones((X_train_scaled.shape[0], 1))
ones_X_test_scaled=np.ones((X_test_scaled.shape[0], 1))

X_train_b=np.c_[ones_X_train_scaled, X_train_scaled]
X_test_b=np.c_[ones_X_test_scaled, X_test_scaled]

m=X_train_b.shape[0]
n=X_train_b.shape[1]
theta=np.random.randn(n, 1)*0.01

eta=0.01
n_iterations=5000
lam = 0.1 

loss_history = []
for i in range(n_iterations):
    pred=X_train_b.dot(theta)
    errors=pred-y_train
    gradients=(2/m)*X_train_b.T.dot(errors)
    reg = 2.0 * lam * theta
    reg[0, 0] = 0.0
    gradients += reg
    theta=theta-(eta*gradients)
    mse = np.sum(errors**2)/m
    loss_history.append(mse)

test_preds = X_test_b.dot(theta)
test_mse = mean_squared_error(y_test, test_preds)
print(f"Gradient Descent Test MSE: {test_mse:.4f}, RMSE: {np.sqrt(test_mse):.4f}")

plt.plot((np.arange(0, len(loss_history)) * 10), loss_history)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient Descent")
plt.show()

sample = X_test[0].reshape(1, -1)
sample_scaled = scaler.transform(sample)
sample_b = np.c_[np.ones((1,1)), sample_scaled]
pred_value = sample_b.dot(theta)
print("Predicted target:", pred_value[0,0])
print("True target:", y_test[0,0])

