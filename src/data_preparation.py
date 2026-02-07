
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

HF_DATASET = "Sudharsanamr/tourism"
HF_OUTPUT  = "Sudharsanamr/tourism"

def main():
    dataset = load_dataset(HF_DATASET)
    df = dataset["train"].to_pandas()

    # Drop ID column
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    # Handle missing values
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)

    processed = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    processed.push_to_hub(HF_OUTPUT)
    print("✅ Data preparation completed and pushed to HF")

if __name__ == "__main__":
    main()
