
import optuna
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from typing import List
from sklearn.metrics import mean_absolute_error

def train_test_lightgbm(ts: pd.DataFrame, shards: List[datetime]) -> List[lgb.basic.Booster, pd.DataFrame]:

    #ts = df_timeseries_gold.copy()

    DROP_COLUMNS = []
    INDEX_COL = 'Timestamp'
    CAT_FEATURES = [
        "Plant",
        "Provider"
    ]
    TARGET = "Vol/Prod_ratio_kg"
    DROP_NAN = False

    ts.drop(columns=DROP_COLUMNS, inplace=True)

    # Convert Providers and Plant Column to binary format
    ts = pd.get_dummies(ts, columns=CAT_FEATURES)

    n_opt_trials = 25
    params = {
        "objective": "regression",
        "metric": "mean_absolute_error",
        "verbosity": -1,
        "boosting_type": "gbdt"
    }

    dfs = []
    for shard in shards:

        test_frame = shard[3].strftime("%Y-%m-%d") + " - " + shard[4].strftime("%Y-%m-%d")
        print("Train-Testing for", shard[3].strftime("%Y-%m-%d") , shard[4].strftime("%Y-%m-%d") )

        train, val, test = (
            ts[ts['Timestamp'] <= shard[0]].copy().drop(columns=[INDEX_COL]),
            ts[(ts['Timestamp'] >= shard[1]) & (ts['Timestamp'] <= shard[2]) ].copy().drop(columns=[INDEX_COL]),
            ts[(ts['Timestamp'] >= shard[3]) & (ts['Timestamp'] <= shard[4]) ].copy(),
        )
        

        train_x = train.drop(columns=[TARGET, "ts_key"])
        train_y = train[TARGET]

        val_x = val.drop(columns=[TARGET, "ts_key"])
        val_y = val.set_index("ts_key")[TARGET]

        test_timestamps = test[INDEX_COL].values
        test = test.drop(columns=[INDEX_COL])
        test_x = test.drop(columns=[TARGET, "ts_key"])
        test_y = test.set_index("ts_key")[TARGET]

        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)

        # callable for optimization
        def objective(trial):
            # Parameters
            params_tuning = {
                "objective": "regression",
                "metric": "mean_absolute_error",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "feature_pre_filter": False,
            }

            gbm = lgb.train(params_tuning, dtrain)
            y_pred = gbm.predict(val_x)
            mae = mean_absolute_error(val_y, y_pred)
            return mae

        # start hyperparameter tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_opt_trials)

        print("Best hyperparameters:", study.best_params)
        print("Best MAE:", study.best_value)

        # Model Training with best params and performance test
        train = ts[ts['Timestamp'] <= shard[2]].copy().drop(columns=['Timestamp'])
        train_x = train.drop(columns=[TARGET, "ts_key"])
        train_y = train[TARGET]
        dtrain = lgb.Dataset(train_x, label=train_y)

        model = lgb.train({**params, **study.best_params}, dtrain)

        # prediction
        y_pred = model.predict(test_x)

        _df_forecast = pd.DataFrame({'Timestamp':test_timestamps, 
                                    'y_true':test_y, 
                                    'y_pred_lgbm':y_pred, 
                                    'test_frame':test_frame})

        dfs.append(_df_forecast)
            
    df_result_lgbm = pd.concat(dfs)

    df_result_lgbm.rename(columns={'y_pred_lgbm': 'LIGHTGBM'}, inplace=True )

    return model, df_result_lgbm