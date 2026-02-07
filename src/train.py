
from datasets import load_dataset
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

HF_DATASET = "Sudharsanamr/tourism"

def main():
    ds = load_dataset(HF_DATASET)
    train_df = ds["train"].to_pandas()

    X = train_df.drop("ProdTaken", axis=1)
    y = train_df["ProdTaken"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    params = {
        "model__n_estimators": [100],
        "model__max_depth": [None, 10],
    }

    grid = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, "models/best_model.joblib")

    print("✅ Model trained and saved")

if __name__ == "__main__":
    main()
