import numpy as np
import pandas as pd

# Classification Metrics
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    ndcg_score,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    max_error,
    explained_variance_score,
)
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp


def generate_classification_report(y_true, y_proba, threshold=0.5):
    valcounts = pd.Series(y_true).value_counts()
    valcounts.sort_index(ascending=True, inplace=True)

    y_pred = (y_proba > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_auc = roc_auc_score(y_true, y_proba)
    report = {
        "Majority Obs ({})".format(valcounts.index[0]): valcounts[0],
        "Minority Obs ({})".format(valcounts.index[1]): valcounts[1],
        "Classification Threshold": threshold,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Gini/Accuracy Ratio": (2 * roc_auc) - 1,
        "ROC_AUC": roc_auc,
        "KS Test_Binned": ks_stat(y_true, y_proba),  # Use custom KS computation (using table)
        "KS Test": ks_2samp(y_true, y_proba).statistic,  # Use scipy
        #         'PSI': "",# calculate_psi(y_true,y_pred,buckets=10),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_true, y_pred),
        "NDCG": ndcg_score([y_true], [y_proba]),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Balanced Accuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2.0,  # (sensitivity + specificity) / 2
    }

    return report


def generate_regression_report(targets, preds):
    reg_metrics = {
        "len": len(targets),
        "r2": r2_score(targets, preds),
        "mape": mean_absolute_percentage_error(targets, preds),
        "mse": mean_squared_error(targets, preds),
        "rmse": mean_squared_error(targets, preds, squared=False),
        "mae": mean_absolute_error(targets, preds),
        #         "msle": mean_squared_log_error(targets, preds), # Check compatible sklearn version
        #         "rmsle": mean_squared_log_error(targets, preds,squared=False),
        "max_error": max_error(targets, preds),
        "explained_variance_score": explained_variance_score(targets, preds),
    }
    return reg_metrics


def generate_clustering_report(X, cluster_ids):

    sample_size = None

    if X.shape[0] > 10000:
        sample_size = 10000  # Use Sampling if number of points > 10k

    return {"silhouette_score": silhouette_score(X, cluster_ids, sample_size=sample_size)}


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


def ks_table(actuals, predictedScores, bins=10):
    # Reference
    # https://github.com/selva86/InformationValue/blob/45654eae92a1d3ba01778a61ef530b8ec8676e1b/R/Main.R#L600

    # sort the actuals and predicred scores and create 10 groups.
    dat = pd.DataFrame({"actuals": actuals, "predictedScores": predictedScores})
    dat = dat.sort_values(by=["predictedScores"], ascending=[False])  # sort desc by predicted scores
    nrows = dat.shape[0]
    rows_in_each_grp = round(dat.shape[0] / bins)
    first_9_grps = np.repeat(range(1, bins), rows_in_each_grp)
    last_grp = np.repeat(bins, nrows - len(first_9_grps))
    grp_index = np.append(first_9_grps, last_grp)
    dat["grp_index"] = grp_index

    # init the ks_table and make the columns.
    ks_tab = (
        dat.groupby("grp_index")
        .agg(total_pop=("actuals", "count"), responders=("actuals", "sum"))
        .reset_index()
        .rename(columns={"grp_index": "rank"})
    )

    ks_tab["non_responders"] = ks_tab.total_pop - ks_tab.responders

    perc_responders_tot = ks_tab.responders.sum() / ks_tab.total_pop.sum()
    ks_tab["expected_responders_by_random"] = ks_tab.total_pop * perc_responders_tot
    ks_tab["perc_responders"] = ks_tab.responders / ks_tab.responders.sum()
    ks_tab["perc_non_responders"] = ks_tab.non_responders / ks_tab.non_responders.sum()

    ks_tab["cum_perc_responders"] = ks_tab.perc_responders.cumsum()
    ks_tab["cum_perc_non_responders"] = ks_tab.perc_non_responders.cumsum()
    ks_tab["difference"] = ks_tab.cum_perc_responders - ks_tab.cum_perc_non_responders
    return ks_tab


def ks_stat(actuals, predictedScores, returnKSTable=False):
    ks_tab = ks_table(actuals, predictedScores)
    if returnKSTable:
        return ks_tab
    else:
        return round(ks_tab.difference.max(), 4)


def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    # https://github.com/mwburke/population-stability-index

    """Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, 
            bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        """

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == "bins":
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == "quantiles":
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            """
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return psi_values

