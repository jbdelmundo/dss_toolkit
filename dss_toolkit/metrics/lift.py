import numpy as np
import pandas as pd


def decile_performance_dataframe(actual, proba, threshold=0.5, bins=10):
    model_perf = pd.DataFrame({"actual": actual, "proba": proba})
    model_perf["prediction"] = (model_perf.proba > threshold).astype(int)

    model_perf["TP"] = ((model_perf.actual == 1) & (model_perf.prediction == 1)).astype(int)
    model_perf["FP"] = ((model_perf.actual == 0) & (model_perf.prediction == 1)).astype(int)
    model_perf["TN"] = ((model_perf.actual == 0) & (model_perf.prediction == 0)).astype(int)
    model_perf["FN"] = ((model_perf.actual == 1) & (model_perf.prediction == 0)).astype(int)

    model_perf.reset_index(inplace=True, drop=True)

    model_perf["decile"] = pd.qcut(model_perf["proba"].rank(method="first"), q=10, labels=False)
    model_perf["decile"] = 10 - model_perf["decile"]  # Inverse (decile 1 contains highest value)

    decile_perf = (
        model_perf.groupby("decile")
        .agg(
            observations=("proba", "count"),
            TP=("TP", sum),
            FP=("FP", sum),
            TN=("TN", sum),
            FN=("FN", sum),
            responders=("actual", sum),
            predicted=("prediction", sum),
            min_score=("proba", min),
            max_score=("proba", max),
            median_score=("proba", "median"),
        )
        .reset_index()
    )

    decile_perf["responders"] = decile_perf["responders"].astype(int)
    decile_perf["non_responders"] = decile_perf.observations - decile_perf.responders
    decile_perf["cum_responders"] = decile_perf.responders.cumsum()
    decile_perf["cum_obs"] = decile_perf.observations.cumsum()
    decile_perf["cum_response_rate"] = decile_perf["cum_responders"] / decile_perf["cum_obs"]

    decile_perf["perc_reseponders"] = decile_perf.responders / decile_perf.responders.sum()
    decile_perf["perc_non_reseponders"] = decile_perf.non_responders / decile_perf.non_responders.sum()
    decile_perf["cum_perc_reseponders"] = decile_perf.perc_reseponders.cumsum()
    decile_perf["cum_perc_non_reseponders"] = decile_perf.perc_non_reseponders.cumsum()
    decile_perf["ks"] = (decile_perf.cum_perc_reseponders - decile_perf.cum_perc_non_reseponders).round(4)

    return decile_perf
