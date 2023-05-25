import numpy as np
import pandas as pd


def decile_performance(actual, proba, threshold=0.5, bins=10):
    model_perf = pd.DataFrame({"actual": actual, "proba": proba})
    model_perf["prediction"] = (model_perf.proba > threshold).astype(int)
    # prediction = model_perf["prediction"].tolist()
    #     model_perf['proba'] = round(model_perf['proba'],1)

    model_perf["TP"] = ((model_perf.actual == 1) & (model_perf.prediction == 1)).astype(int)
    model_perf["FP"] = ((model_perf.actual == 0) & (model_perf.prediction == 1)).astype(int)
    model_perf["TN"] = ((model_perf.actual == 0) & (model_perf.prediction == 0)).astype(int)
    model_perf["FN"] = ((model_perf.actual == 1) & (model_perf.prediction == 0)).astype(int)

    model_perf.reset_index(inplace=True)
    model_perf.drop(columns="index", inplace=True)

    # Assgin decile (1 is highest) # pd.qcut doesnt divide equally on ties
    model_perf_sorted = model_perf.sort_values("proba", ascending=False)
    count_per_bin = int(model_perf.shape[0] / bins)
    first_9 = np.repeat(range(1, bins), count_per_bin)
    model_perf_sorted["decile"] = np.hstack([first_9, np.repeat(bins, model_perf.shape[0] - first_9.shape[0])])

    model_perf = model_perf_sorted.sort_index()  # Return original ordering

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

    decile_perf["non_responders"] = decile_perf.observations - decile_perf.responders
    decile_perf["cum_responders"] = decile_perf.responders.cumsum()
    decile_perf["cum_obs"] = decile_perf.observations.cumsum()
    decile_perf["cum_response_rate"] = decile_perf["cum_responders"] / decile_perf["cum_obs"]

    decile_perf["perc_reseponders"] = decile_perf.responders / decile_perf.responders.sum()
    decile_perf["perc_non_reseponders"] = decile_perf.non_responders / decile_perf.non_responders.sum()
    decile_perf["cum_perc_reseponders"] = decile_perf.perc_reseponders.cumsum()
    decile_perf["cum_perc_non_reseponders"] = decile_perf.perc_non_reseponders.cumsum()
    decile_perf["ks"] = decile_perf.cum_perc_reseponders - decile_perf.cum_perc_non_reseponders

    return decile_perf
