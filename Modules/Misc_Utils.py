import pandas as pd
import numpy as np


def get_average_score(comparator_output, metric_col_name: str, group_col_name: str = "Fingerprint", normalize_scores: bool = False):
    comparator_output_mean: pd.DataFrame = comparator_output.copy()  # Copy the comparator_output dataframe
    mean_score: float = comparator_output_mean[f"{metric_col_name}"].mean()  # Calculate the mean score in the metric_col_name column

    ### Normalize the scores

    if normalize_scores is False:
        pass
    else:
        # for i in range(len(comparator_output)):
        # comparator_output_mean.loc[i, f"{metric_col_name}"] = np.mean(comparator_output.loc[i, f"{metric_col_name}"])
        # comparator_output_mean.loc[i, f"{metric_col_name}"] = comparator_output_mean.loc[i, f"{metric_col_name}"] - mean_score

        # comparator_output_mean[f"{metric_col_name}"] = comparator_output_mean[f"{metric_col_name}"].mean()
        comparator_output_mean[f"{metric_col_name}"] = comparator_output_mean[f"{metric_col_name}"].subtract(mean_score)

    groupby_col_and_mean = comparator_output_mean.groupby([f"{group_col_name}"]).mean()

    comparator_output_mean[f"{group_col_name}_Group_Mean"] = comparator_output_mean[f"{group_col_name}"].map(
        groupby_col_and_mean[f"{metric_col_name}"]
    )

    scores_df = comparator_output_mean.sort_values(f"{group_col_name}_Group_Mean", ascending=False)

    return scores_df, mean_score
