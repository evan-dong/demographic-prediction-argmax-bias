import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

### MISC HELPERS


def preds_to_marginal(predictions, n_classes: int | None = None):
    return np.bincount(predictions, minlength=n_classes)


def posterior_to_aggregate(posterior):
    return np.sum(posterior, axis=0).astype(float)


def exact_counts(frequencies, size: int):
    """
    given a probability distribution and a size, give the closest discrete marginal distribution of that size.
    I believe this is closest by L1 distance - we're just flooring the decimal components for each dimension and
    then filling in the remaining values needed by highest probability. that may not strictly be closest by other measures?
    """
    ## discretize to exact total size of batch_size
    reference_dist = frequencies * size
    ## all the points that are  definitely there
    floored = np.floor(reference_dist).astype(int)
    ## how do we allocate the leftover fractions?
    rounding_values = reference_dist - floored
    rounding = size - np.sum(floored)
    ## if we have to choose between the fractions to round up, choose the largest ones
    plus_one = (
        np.argsort(rounding_values)[
            -rounding.astype(int) :  ## ascending, so take the largest
        ]
        if rounding > 0
        else []
    )  ## if you have exactly 0, there's a bug!
    added = np.zeros(rounding_values.shape[0], dtype=int)
    added[plus_one] = 1
    reference_dist = floored + added
    return reference_dist


def generate_reference_dist(labels, batch_size: int, min_classes: int | None = None):
    """
    rescale the true label frequencies to the (L1-)closest discrete distribution of the batch size
    """
    true_counts = np.bincount(labels, minlength=min_classes)
    true_frequencies = true_counts / labels.shape[0]
    assert np.isclose(np.sum(true_counts), labels.shape[0]).all(), np.sum(true_counts)
    reference_dist = exact_counts(true_frequencies, batch_size)
    return reference_dist


def conditional_entropy(joint_probs):
    p_x = np.sum(joint_probs, axis=0)
    cond_prob = joint_probs / p_x
    cond_entropy = -np.sum(joint_probs * np.log(cond_prob))
    return cond_entropy


def batch_dataset(probs, batch_size: int, remainder: bool = False):
    """
    probs is (n_samples, n_classes)
    """
    if remainder:
        ## make the batches an exact size and then make a remainder batch
        n_full_batches = probs.shape[0] // batch_size
        partial_size = probs.shape[0] % n_full_batches
        batches = np.split(probs[:-partial_size], n_full_batches)
        batches.append(probs[-partial_size:])
    else:
        ## divide as evenly as possible
        n_batches = max(
            np.round(probs.shape[0] / batch_size), 1
        )  ## whatever's closer to the intended size
        batches = np.array_split(probs, n_batches)

    return batches


def conditional_batching(
    df: pd.DataFrame,
    batch_size: int,
    condition_columns: list[str],
    prob_cols: list[str],
    marginal_label_col: str | None = None,
    remainder=False,
) -> tuple[list, list]:
    """
    calculates true marginal max weight matching if given marginal label column
    marginal_label_col (str | None): the name of column
    """

    grouped = (
        [("all", df)] if condition_columns is None else df.groupby(condition_columns)
    )
    batches_by_condition = []
    references_by_condition = []
    for condition, subset in grouped:
        print("conditional value", condition)
        print("condition size", len(subset))
        probs = subset[prob_cols]

        batches = batch_dataset(probs, batch_size, remainder=remainder)

        if marginal_label_col is not None:  # true marginal
            reference_dist = (
                np.bincount(subset[marginal_label_col], minlength=len(prob_cols))
                / subset.shape[0]
            )
            references = [reference_dist for _ in batches]
        else:
            references = [np.mean(batch, axis=0) for batch in batches]

        batches_by_condition.append(batches)
        ## these references *could* go unrepeated
        references_by_condition.append(references)

    all_batches = [batch for b in batches_by_condition for batch in b]

    all_references = [ref for r in references_by_condition for ref in r]

    return all_batches, all_references


def preds_to_counts(predictions_df: pd.DataFrame, index_map=None):
    """
    ALL COLUMNS MUST BE OF THE SAME TYPE
    """
    counts = (
        pd.concat(
            [
                predictions_df[col].value_counts(sort=False)
                for col in predictions_df.columns
            ],
            axis=1,
        )
        .sort_index()
        .fillna(0)
        .dropna(how="all")
    )
    counts.columns = predictions_df.columns
    ## is there a way we can easily set how many values there must be?
    if index_map is not None:
        counts.index = counts.index.map(index_map)
    return counts


def recalibrate_posterior(posterior, true_labels):
    """for a posterior of shape (n_samples, n_classes)"""

    ## logistic regression is equivalent to Platt scaling
    calibrated_model = LogisticRegression(solver="saga", penalty=None).fit(
        posterior, true_labels
    )
    calibrated_probs = calibrated_model.predict_proba(posterior)
    return calibrated_probs
