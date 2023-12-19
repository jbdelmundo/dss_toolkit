import numpy as np
import pandas as pd

# Classification Metrics
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from scipy.stats import ks_2samp


def generate_report(y_true, y_proba, threshold=0.5):
    valcounts = pd.Series(y_true).value_counts()
    valcounts.sort_index(ascending=True, inplace=True)

    y_pred = (y_proba > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_auc = roc_auc_score(y_true, y_proba)
    report = {
        "Total Obs": y_true.shape[0],
        "Majority Obs ({})".format(valcounts.index[0]): valcounts[0],
        "Minority Obs ({})".format(valcounts.index[1]): valcounts[1],
        "Classification Threshold": threshold,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Gini/Accuracy Ratio": (2 * roc_auc) - 1,
        "ROC_AUC": roc_auc,
        "KS Separation": ks_stat(
            y_true, y_proba
        ),  # Use custom KS computation (using table)
        "KS Statistic": ks_2samp(y_true, y_proba).statistic,  # Use scipy
        #         'PSI': "",# calculate_psi(y_true,y_pred,buckets=10),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_true, y_pred),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Balanced Accuracy": ((tp / (tp + fn)) + (tn / (tn + fp)))
        / 2.0,  # (sensitivity + specificity) / 2
    }

    return report


def ks_table(actuals, predictedScores, bins=10):
    # Reference
    # https://github.com/selva86/InformationValue/blob/45654eae92a1d3ba01778a61ef530b8ec8676e1b/R/Main.R#L600

    # sort the actuals and predicred scores and create 10 groups.
    dat = pd.DataFrame({"actuals": actuals, "predictedScores": predictedScores})
    dat = dat.sort_values(
        by=["predictedScores"], ascending=[False]
    )  # sort desc by predicted scores
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
