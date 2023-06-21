from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def smote_r(train_data, train_labels, perc_over, perc_under, k=5):
    """
    Recoded to match the parameters of SMOTE in R
    https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/SMOTE

    perc_over: percent increase in minority
    perc_under: new count of majority = (increase in minority) * perc_under

    """

    label_count = train_labels.value_counts().sort_values(ascending=False)
    majority = label_count[0]
    minority = label_count[1]

    new_minority = int(minority * (1 + perc_over))
    new_majority = int((new_minority - minority) * perc_under)

    over = SMOTE(sampling_strategy={1: new_minority}, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy={0: new_majority})
    smote_pipeline = ImbPipeline(steps=[("o", over), ("u", under)])

    X_resampled, y_resampled = smote_pipeline.fit_resample(train_data, train_labels)
    return X_resampled, y_resampled


def smote_py(train_data, train_labels, oversample, undersample, k=5):
    steps = []
    if oversample is not None:
        over = SMOTE(sampling_strategy=oversample, k_neighbors=k)
        steps.append(("o", over))

    if undersample is not None:
        under = RandomUnderSampler(sampling_strategy=undersample)
        steps.append(("u", under))

    if oversample is None and undersample is None:
        return train_data, train_labels  # Do nothing

    smote_pipeline = ImbPipeline(steps=steps)

    X_resampled, y_resampled = smote_pipeline.fit_resample(train_data, train_labels)
    return X_resampled, y_resampled
