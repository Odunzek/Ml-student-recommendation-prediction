"""
Create pipeline with MLflow tracking and logging
This version properly logs the model to MLflow for version control and tracking
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

print("=" * 60)
print("CREATING PIPELINE WITH MLFLOW TRACKING")
print("=" * 60)

# Set MLflow experiment
mlflow.set_experiment("student_recommendation_system")

# Start MLflow run
with mlflow.start_run(run_name="XGBoost_Regression_Pipeline") as run:

    # Load and preprocess data (same as before)
    print("\n1. Loading data...")
    df = pd.read_csv("../data/StudentPerformanceFactors.csv")

    # Handle missing values
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna('Unknown')
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna('Unknown')

    # Generate synthetic failing students
    print("2. Generating synthetic failing students...")
    n_fail = 1300

    def sample(col):
        return np.random.choice(df[col].unique(), size=n_fail, p=df[col].value_counts(normalize=True))

    synthetic = pd.DataFrame({
        "Hours_Studied": np.random.normal(df["Hours_Studied"].mean() - 6, 4, n_fail).clip(0),
        "Attendance": np.random.normal(df["Attendance"].mean() - 12, 8, n_fail).clip(40, 100),
        "Parental_Involvement": sample("Parental_Involvement"),
        "Access_to_Resources": sample("Access_to_Resources"),
        "Extracurricular_Activities": sample("Extracurricular_Activities"),
        "Sleep_Hours": np.random.normal(df["Sleep_Hours"].mean() - 1.5, 1, n_fail).clip(3, 10),
        "Previous_Scores": np.random.normal(df["Previous_Scores"].mean() - 10, 12, n_fail).clip(20, 75),
        "Motivation_Level": sample("Motivation_Level"),
        "Internet_Access": sample("Internet_Access"),
        "Tutoring_Sessions": np.random.poisson(df["Tutoring_Sessions"].mean() - 1, n_fail).clip(0),
        "Family_Income": sample("Family_Income"),
        "Teacher_Quality": sample("Teacher_Quality"),
        "School_Type": sample("School_Type"),
        "Peer_Influence": sample("Peer_Influence"),
        "Physical_Activity": np.random.normal(df["Physical_Activity"].mean() - 1, 1.5, n_fail).clip(0),
        "Learning_Disabilities": sample("Learning_Disabilities"),
        "Distance_from_Home": sample("Distance_from_Home"),
        "Gender": sample("Gender"),
    })
    synthetic["Exam_Score"] = np.random.randint(20, 56, n_fail)

    df = pd.concat([df, synthetic], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Log data info
    mlflow.log_param("original_samples", 6607)
    mlflow.log_param("synthetic_samples", n_fail)
    mlflow.log_param("total_samples", len(df))

    # Encode categorical variables
    print("3. Encoding categorical variables...")
    lmh = {'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 0}
    df['Parental_Involvement'] = df['Parental_Involvement'].map(lmh)
    df['Access_to_Resources'] = df['Access_to_Resources'].map(lmh)
    df['Teacher_Quality'] = df['Teacher_Quality'].map(lmh)
    df['Family_Income'] = df['Family_Income'].map(lmh)
    df['Motivation_Level'] = df['Motivation_Level'].map(lmh)

    df['Distance_from_Home'] = df['Distance_from_Home'].map({'Near': 1, 'Moderate': 2, 'Far': 3, 'Unknown': 0})
    df['Peer_Influence'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 2, 'Positive': 3})

    bin_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    bin_cols = ['Extracurricular_Activities','Internet_Access','Learning_Disabilities','Gender']
    for c in bin_cols:
        df[c] = df[c].map(bin_map)

    df['School_Type'] = df['School_Type'].map({'Public': 0, 'Private': 1})
    df = pd.get_dummies(df, columns=['Parental_Education_Level'], drop_first=True)

    bool_cols = ['Parental_Education_Level_High School', 'Parental_Education_Level_Postgraduate', 'Parental_Education_Level_Unknown']
    df[bool_cols] = df[bool_cols].astype(int)

    # Split data
    print("4. Splitting data...")
    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log split info
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Define numeric vs categorical columns
    numeric_cols = X.select_dtypes(include=["float64"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    print(f"\n   Numeric columns (will be scaled): {len(numeric_cols)}")
    print(f"   Categorical columns (passthrough): {len(categorical_cols)}")

    mlflow.log_param("numeric_features", len(numeric_cols))
    mlflow.log_param("categorical_features", len(categorical_cols))

    # Create pipeline
    print("\n5. Creating pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', 'passthrough', categorical_cols)
        ],
        remainder='passthrough'
    )

    # Log model hyperparameters
    model_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42
    }

    for param, value in model_params.items():
        mlflow.log_param(param, value)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(**model_params, verbosity=0))
    ])

    # Train pipeline
    print("6. Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("7. Evaluating pipeline...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Log metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)

    print(f"\n   Train MAE: {train_mae:.3f}")
    print(f"   Test MAE:  {test_mae:.3f}")
    print(f"   Train R2:  {train_r2:.3f}")
    print(f"   Test R2:   {test_r2:.3f}")

    # Log the pipeline to MLflow
    print("\n8. Logging pipeline to MLflow...")
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name="StudentExamScorePredictor"
    )

    # Also save locally with pickle (for backward compatibility)
    save_path = "../dashboard/student_regression_pipeline.pkl"
    print(f"9. Saving pipeline locally to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Print MLflow info
    run_id = run.info.run_id
    print("\n" + "="*60)
    print("[SUCCESS] PIPELINE CREATED AND LOGGED TO MLFLOW!")
    print("="*60)
    print(f"   Run ID: {run_id}")
    print(f"   Model URI: runs:/{run_id}/model")
    print(f"   Local file: {save_path}")
    print(f"   Test MAE: {test_mae:.3f}")
    print(f"   Test R2:  {test_r2:.3f}")
    print("="*60)
    print("\nView in MLflow UI:")
    print("   cd project/notebooks")
    print("   mlflow ui")
    print("   Then open: http://localhost:5000")
    print("="*60)
