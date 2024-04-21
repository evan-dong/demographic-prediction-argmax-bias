import numpy as np
import scipy as sp
from sklearn.metrics import accuracy_score
from utils import exact_counts, preds_to_counts
import pandas as pd
from typing import Literal

FIDELITY_METRICS = Literal["L1", "L2", "KL"]


def true_accuracy(true_labels, predictions):
    return accuracy_score(true_labels, predictions)


def mean_expected_accuracy(posterior, predictions):
    return np.mean(np.array(posterior)[np.arange(posterior.shape[0]), predictions])


def distribution_fidelity(reference, counts, metric: FIDELITY_METRICS = "L1"):
    """takes in two distributions, normalizes each to probabilities/unit vectors, and calculates the approporiate fidelity metric"""
    match metric:
        case "L1":
            return -np.linalg.norm(
                reference / np.sum(reference) - counts / np.sum(counts), 1
            )
        case "L2":
            return -np.linalg.norm(
                reference / np.sum(reference) - counts / np.sum(counts), 2
            )
        case "KL":
            return -np.mean(sp.rel_entr(counts, reference))
        case _:
            raise ValueError(
                "Not an implemented marginal objective. Please use one of {'L2', 'L1', 'KL'}"
            )


def assess_predictions(
    preds,
    fidelity_metric: FIDELITY_METRICS,
    long_labels=None,
    posterior=None,
    prior=None,
):
    """
    preds (pd.DataFrame): is (n_samples, n_methods) where each column corresponds to a particular set of predictions by a particular method.
    "Ground Truth Fidelity" is distance to the true sampled labels' marginal
    "Aggregate Posterior Fidelity" is distance to the sample's aggregate posterior (mirroring the posterior being "Expected Accuracy")
    "Prior Distribution Fidelity" is distance to the prior
    """
    n_classes = int(np.max(long_labels) + 1)

    ## can also reduce repeated bincounts
    cols = preds.columns
    preds = np.array(preds)
    bincounted = np.array(
        [np.bincount(col, minlength=n_classes) for col in preds.T]
    )  ## (n_methods, n_classes)

    results = pd.DataFrame(index=cols)
    if posterior is not None:
        aggregate_posterior = np.sum(posterior, axis=0)
        results["Expected Accuracy"] = np.mean(
            np.array(posterior)[np.arange(len(long_labels)), preds.T], axis=1
        )
        results["Aggregate Posterior Fidelity"] = [
            distribution_fidelity(
                aggregate_posterior,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]

    if long_labels is not None:
        true_label_counts = np.bincount(long_labels, minlength=n_classes)
        results["Accuracy"] = np.mean(
            preds == np.array(long_labels)[:, np.newaxis], axis=0
        )
        results["Ground Truth Fidelity"] = [
            distribution_fidelity(
                true_label_counts,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]

    if prior is not None:
        results["Prior Distribution Fidelity"] = [
            distribution_fidelity(
                prior,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]
    # results = pd.DataFrame(results, index=cols)

    return results


def ragged_assess(
    preds,
    fidelity_metric,
    uncoded_int: int = 6,
    long_labels=None,
    posterior=None,
    prior=None,
):
    """
    preds (pd.DataFrame): is int (n_samples, n_methods) where each column corresponds to a particular set of predictions by a particular method.
    use_masks is list (or numpy array, or whatever) of (n_methods, n_samples) of what values should be used
    "Ground Truth Fidelity" is distance to the true sampled labels' marginal
    "Aggregate Posterior Fidelity" is distance to the sample's aggregate posterior (mirroring the posterior being "Expected Accuracy")
    "Prior Distribution Fidelity" is distance to the prior
    """
    cols = preds.columns
    preds = preds.astype(int)

    bincounted = np.array(preds_to_counts(preds).drop(labels=uncoded_int)).T
    method_sums = np.sum(bincounted, axis=1)  ##
    preds = np.array(preds)  ## n_samples, n_methods

    assert (preds <= uncoded_int).all()  ## needs to be this way

    results = pd.DataFrame(index=cols)
    results["Dropped Fraction"] = 1 - (method_sums / preds.shape[0])

    if posterior is not None:
        assert (
            uncoded_int > posterior.shape[1] - 1
        )  ## uncoded should be outside the posterior prob distribution, by definition.
        reshaped_posterior = np.zeros(
            (posterior.shape[0], uncoded_int + 1)
        )  ## +1 so that you can index into it
        reshaped_posterior[: posterior.shape[0], : posterior.shape[1]] = posterior

        results["Expected Accuracy"] = (
            np.sum(reshaped_posterior[np.arange(len(long_labels)), preds.T], axis=1)
            / method_sums
        )

        aggregate_posterior = np.array(np.sum(posterior, axis=0))

        results["Aggregate Posterior Fidelity"] = [
            distribution_fidelity(
                aggregate_posterior,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]

    if long_labels is not None:
        results["Accuracy"] = (
            np.sum(preds == np.array(long_labels)[:, np.newaxis], axis=0) / method_sums
        )

        true_label_counts = np.bincount(long_labels)
        results["Ground Truth Fidelity"] = [
            distribution_fidelity(
                true_label_counts,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]

    if prior is not None:
        results["Prior Distribution Fidelity"] = [
            distribution_fidelity(
                prior,
                counts,
                metric=fidelity_metric,
            )
            for counts in bincounted
        ]

    return results


def calculate_batched_fidelity(batches):
    """
    how much does aggregate posterior fidelity differ when batching from a global joint optimization?
    """
    total_marginal = 0

    if remainder:
        n_full_batches = probs.shape[0] // batch_size
        partial_size = probs.shape[0] % n_full_batches
        batches = np.split(probs[:-partial_size], n_full_batches)
        for batch in batches:
            total_marginal += exact_counts(np.mean(batch, axis=0), batch_size)
        total_marginal += exact_counts(
            np.mean(probs[-partial_size:], axis=0), partial_size
        )

    else:
        ## divide as evenly as possible
        n_batches = np.round(
            probs.shape[0] / batch_size
        )  ## whatever's closer to the intended size
        batches = np.array_split(probs, n_batches)
        for batch in batches:
            total_marginal += exact_counts(np.mean(batch, axis=0), batch.shape[0])
    return total_marginal
