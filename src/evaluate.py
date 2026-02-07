
from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

HF_DATASET = "Sudharsanamr/tourism"

def main():
    ds = load_dataset(HF_DATASET)
    test_df = ds["test"].to_pandas()

    X_test = test_df.drop("ProdTaken", axis=1)
    y_test = test_df["ProdTaken"]

    model = joblib.load("models/best_model.joblib")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy :", accuracy_score(y_test, preds))
    print("ROC-AUC  :", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    main()
