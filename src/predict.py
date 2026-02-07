
import joblib
import pandas as pd

def main():
    model = joblib.load("models/best_model.joblib")

    sample = {
        "Age": 35,
        "TypeofContact": "Self Inquiry",
        "CityTier": 1,
        "Occupation": "Salaried",
        "Gender": "Male",
        "NumberOfPersonVisiting": 2,
        "PreferredPropertyStar": 4,
        "MaritalStatus": "Married",
        "NumberOfTrips": 3,
        "Passport": 1,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 0,
        "Designation": "Manager",
        "MonthlyIncome": 60000,
        "PitchSatisfactionScore": 4,
        "ProductPitched": "Wellness",
        "NumberOfFollowups": 2,
        "DurationOfPitch": 20
    }

    df = pd.DataFrame([sample])
    pred = model.predict(df)[0]

    print("Prediction successful:", pred)

if __name__ == "__main__":
    main()
