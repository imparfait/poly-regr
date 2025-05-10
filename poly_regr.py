import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("fuel_consumption_vs_speed.csv")
X = df["speed_kmh"].values.reshape(-1, 1) 
y = df["fuel_consumption_l_per_100km"].values
mse_list = []
mae_list = []
degrees = range(1, 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_list.append(mse)
    mae_list.append(mae)
    print(f"Ступінь {degree}: MSE = {mse:.4f}, MAE = {mae:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(degrees, mse_list, marker='o', label='MSE')
plt.plot(degrees, mae_list, marker='s', label='MAE')
plt.xlabel("Ступінь полінома")
plt.ylabel("Значення метрик")
plt.title("Оцінка точності моделей")
plt.legend()
plt.grid(True)
plt.show()

best_degree = degrees[np.argmin(mse_list)]  
poly_best = PolynomialFeatures(best_degree)
X_poly = poly_best.fit_transform(X)
model_best = LinearRegression()
model_best.fit(X_poly, y)
X_range = np.linspace(20, 140, 300).reshape(-1, 1)
X_range_poly = poly_best.transform(X_range)
y_range_pred = model_best.predict(X_range_poly)
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Виміряні дані')
plt.plot(X_range, y_range_pred, color='red', label=f'Поліном ступеня {best_degree}')
plt.xlabel("Швидкість")
plt.ylabel("Витрати пального")
plt.title("Поліноміальна регресія")
plt.legend()
plt.grid(True)
plt.show()

X_predict = np.array([35, 95, 140]).reshape(-1, 1)
X_predict_poly = poly_best.transform(X_predict)
y_predict = model_best.predict(X_predict_poly)
for speed, consumption in zip(X_predict.ravel(), y_predict):
    print(f" {speed} ≈ {consumption:.2f} ")