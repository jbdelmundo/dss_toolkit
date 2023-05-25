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

from scipy.stats import ks_2samp
from ks_test import ks_stat


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

# TODO
# multiclass_classification report
