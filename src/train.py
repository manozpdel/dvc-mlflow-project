import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/features/data.csv")

X = df[["X", "X_squared"]]
y = df["y"]

mlflow.set_experiment("dvc-mlflow-demo")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(model, "model")

    print("Training complete, MSE:", mse)