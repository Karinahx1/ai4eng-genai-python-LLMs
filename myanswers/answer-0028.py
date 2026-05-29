import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def segment_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:

    features = [
        "age",
        "income_usd",
        "visits_per_month",
        "avg_basket_usd",
        "loyalty_years",
    ]

    target = "monthly_spend_usd"

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(df_train[features])
    X_train_sc = scaler.fit_transform(X_train_imp)

    X_test_imp = imputer.transform(df_test[features])
    X_test_sc = scaler.transform(X_test_imp)

    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    train_segments = km.fit_predict(X_train_sc)
    test_segments = km.predict(X_test_sc)

    predictions = np.zeros(len(df_test))
    rmse_per_seg = {}
    segment_sizes = {}

    for seg in range(3):

        tr_idx = np.where(train_segments == seg)[0]
        te_idx = np.where(test_segments == seg)[0]

        segment_sizes[seg] = len(tr_idx)

        if len(te_idx) == 0:
            rmse_per_seg[seg] = None
            continue

        model = Ridge(alpha=10)

        model.fit(
            X_train_sc[tr_idx],
            df_train[target].iloc[tr_idx]
        )

        preds_seg = model.predict(X_test_sc[te_idx])

        predictions[te_idx] = preds_seg

        rmse_per_seg[seg] = np.sqrt(
            mean_squared_error(
                df_test[target].iloc[te_idx],
                preds_seg
            )
        )

    return {
        "segments_test": test_segments,
        "predictions": predictions,
        "rmse_per_segment": rmse_per_seg,
        "segment_sizes": segment_sizes,
    }