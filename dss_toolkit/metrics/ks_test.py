import numpy as np
import pandas as pd


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
