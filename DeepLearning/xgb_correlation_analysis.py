import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


CSV_PATH = Path(__file__).resolve().parent / "xgb_demo_data.csv"

FEATURES = [
    "MB_IPB_VehLongAccel",
    "MB_IPB_VehLateralAccel",
    "MB_IPB_YAWRate",
    "Monr_AI_f32_SteerAgFA_rad",
    "Monr_AI_f32_RoadSteerAgRA_rad",
    "Monr_AI_f32_VehSpdLgtEstim_mps",
    "Monr_AI_b_ABSActv_flg",
    "Monr_AI_b_TCSActv_flg",
    "Monr_AI_b_VDCActv_flg",
]
TARGET = "VelForward"


def main() -> None:
    # 1) Load data
    df = pd.read_csv(CSV_PATH, usecols=FEATURES + [TARGET], low_memory=False).dropna()
    print(f"Data rows: {len(df)}, features: {len(FEATURES)}")

    # 2) Quick linear-correlation check (Pearson)
    corr = (
        df[FEATURES + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    print("\n=== Pearson correlation with target (abs sorted) ===")
    print(corr)

    # 3) Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=42,
    )

    # 4) XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5) Validation metrics
    pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, pred) ** 0.5
    mae = mean_absolute_error(y_valid, pred)
    r2 = r2_score(y_valid, pred)

    print("\n=== Validation metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    # 6) Model-based importance
    gain_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "XGB_importance": model.feature_importances_,
        }
    ).sort_values("XGB_importance", ascending=False)

    print("\n=== XGBoost built-in feature importance ===")
    print(gain_importance.to_string(index=False))

    # 7) Permutation importance on validation set (more robust than split count)
    perm = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=10,
        random_state=42,
        scoring="r2",
    )
    perm_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Permutation_mean": perm.importances_mean,
            "Permutation_std": perm.importances_std,
        }
    ).sort_values("Permutation_mean", ascending=False)

    print("\n=== Permutation importance (on validation, scoring=R2) ===")
    print(perm_df.to_string(index=False))


if __name__ == "__main__":
    main()
