import numpy as np
import scipy.special as sp
import cvxpy as cp
import scipy.optimize as opt
from utils import exact_counts, batch_dataset
import pandas as pd
from multiprocessing import Pool
import os
from metrics import FIDELITY_METRICS
from sklearn import svm

### BASIC DECISION FUNCTIONS
# region


##### Individual Decision Functions
# region
def mode_discretization(matrix):
    return np.argmax(matrix, axis=1)


def thompson_discretization(matrix):
    matrix = np.array(matrix)
    ## vectorized
    randchoice = np.random.random_sample(size=matrix.shape[0])
    cumprobs = np.cumsum(matrix, axis=1)
    answers = np.sum(randchoice[:, np.newaxis] > cumprobs, axis=1)
    return np.array(answers)


def nucleus_sampling(probs, top_k: int | None = None):
    probs = np.array(probs)
    n_units, n_classes = probs.shape
    bottom_k = n_classes - top_k
    if (2 > top_k) or (top_k > n_classes - 1):
        raise ValueError("top_k must be between (n_classes-1) and 2, inclusive")
    probs = np.array(probs).copy()

    ## in case of ties, keep both tied classes
    boundary_vals = np.partition(probs, bottom_k)[:, bottom_k]
    dropped = probs < boundary_vals[:, np.newaxis]
    probs[dropped] = 0
    probs /= np.sum(probs, axis=1)[:, np.newaxis]

    return thompson_discretization(probs)


# endregion

##### Joint Decision Functions
# region


def max_weight_discretization(
    matrix, reference_distribution=None, capacity_factor=1
) -> np.ndarray:
    """
    We want to use opt.linear_sum_assignment to find the optimal assignment. However, need to do some bookkeeping where we repeat labels the appropriate number of times.
    """
    ## n_classes
    number_of_candidates = matrix.shape[1]
    assert np.isclose(np.sum(np.array(matrix)), matrix.shape[0])
    ## max number of data points that can be in one class
    num_for_each_index = (
        exact_counts(np.mean(matrix, axis=0) * capacity_factor, matrix.shape[0])
        if reference_distribution is None
        else exact_counts(
            reference_distribution / np.sum(reference_distribution), matrix.shape[0]
        )
    )  ## you can remove this/precalculate it if you really want, strictly speaking
    print("max counts allocated per class", num_for_each_index)

    assert len(num_for_each_index) == number_of_candidates

    ## duplicate the probabilities some n number of times
    matrix_full = np.repeat(matrix, num_for_each_index, axis=1)
    ### THIS CAN BE SPEEDED UP IN THEORY
    ## calculate and pull out optimal indices
    answer_full = opt.linear_sum_assignment(matrix_full, maximize=True)[1]
    answer_orig = np.digitize(answer_full, np.cumsum(num_for_each_index))

    assert len(answer_orig) == matrix.shape[0], (len(answer_orig), matrix_full.shape)

    return answer_orig


def balance_objectives(
    posterior,
    gamma: float,
    marginal_objective: FIDELITY_METRICS,
    calibration_distribution=None,
    solver=cp.GUROBI,
    timeout: int = 30,
    verbose: bool = False,
) -> np.ndarray:
    """ """
    ## aggregate posterior if not otherwise specified
    calibration_distribution = (
        np.sum(posterior, axis=0) / posterior.shape[0]
        if calibration_distribution is None
        else calibration_distribution
    )

    training_size = posterior.shape[0]
    n_classes = posterior.shape[1]

    # initialize variables
    class_assignment = cp.Variable(
        (training_size, n_classes), boolean=True
    )  # nonneg=True) ## add in binary somewhere?
    acc_objective = cp.sum(cp.multiply(class_assignment, posterior)) / training_size

    marginal_distribution = (
        cp.sum(class_assignment, axis=0) / training_size
    )  ## normalize to one

    match marginal_objective:
        case "L1":
            distribution_objective = cp.norm(
                marginal_distribution - calibration_distribution, 1
            )
        case "L2":
            distribution_objective = cp.norm(
                marginal_distribution - calibration_distribution, 2
            )
        case "KL":
            distribution_objective = cp.sum(
                cp.rel_entr(marginal_distribution, calibration_distribution)
            )
        case _:
            raise ValueError(
                "Not an implemented marginal objective. Please use one of {'L2', 'L1', 'KL'}"
            )

    ## maximize accuracy (linear) and minimize {L1 distnce, L2 distance, KL divergence} (convex)
    total_objective = gamma * acc_objective - (1 - gamma) * distribution_objective

    constraints = [
        cp.sum(class_assignment, axis=1) == 1,
    ]

    objective = cp.Maximize(total_objective)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver, verbose=verbose, TimeLimit=timeout)

    predicted_labels = np.argmax(class_assignment.value, axis=1)
    if verbose:
        print("gamma: ", gamma)
        print("integer program expected accuracy: ", acc_objective.value)
        print(f"integer program {marginal_objective}: ", distribution_objective.value)
    return predicted_labels


# endregion
# endregion

### MULTIPLE SETS OF DECISIONS


def pareto_curve_sweep(
    posterior,
    gammas,
    fidelity_metric: FIDELITY_METRICS,
    timeout: int = 10,
    n_processes: int = 0,
) -> list[np.ndarray]:
    """
    calculates integer program for multiple gammas across one posterior.
    returns a list[np.array] of shape (n_gammas, n_samples)
    """
    n_processes = os.cpu_count() if n_processes == 0 else n_processes  ## interpret

    if n_processes > 1:
        reiterate = [
            (posterior, gamma, fidelity_metric, None, cp.GUROBI, timeout)
            for gamma in gammas
        ]
        with Pool(n_processes) as p:
            integer_program_preds = p.starmap(
                balance_objectives, reiterate, chunksize=1
            )
    else:
        integer_program_preds = [
            balance_objectives(
                posterior, gamma, marginal_objective=fidelity_metric, timeout=timeout
            )
            for gamma in gammas
        ]
    return integer_program_preds


### Batching
# region
def batched_integer_program(
    batches: list[pd.DataFrame],
    gammas,
    fidelity_metric: FIDELITY_METRICS,
    timeout: int = 10,
    n_processes: int = 0,
) -> list[pd.Series]:
    """
    calculates integer program across multiple batches for multiple gammas. each batch can be different sizes.
    returns of list[pd.Series] of shape (n_gammas, n_samples)
    """
    n_processes = os.cpu_count() if n_processes == 0 else n_processes  ## interpret

    full_index = pd.concat(batches).index

    if n_processes > 1:
        ## n_gammas, n_batches
        reiterate = [
            [
                (batch, gamma, fidelity_metric, None, cp.GUROBI, timeout)
                for batch in batches
            ]
            for gamma in gammas
        ]  ##[batch1gamma1, batch2gamma1, ..., batch1gamma2, ...]
        with Pool(n_processes) as p:
            integer_program_preds = [
                np.concatenate(p.starmap(balance_objectives, gamma_chunk, chunksize=1))
                for gamma_chunk in reiterate
            ]
    else:
        integer_program_preds = []
        for gamma in gammas:
            batched_prediction = [
                balance_objectives(
                    batch, gamma, marginal_objective=fidelity_metric, timeout=timeout
                )
                for batch in batches
            ]
            integer_program_preds.append(np.concatenate(batched_prediction))

    integer_program_preds = [
        pd.Series(prediction, index=full_index) for prediction in integer_program_preds
    ]
    return integer_program_preds


def batched_max_weight(
    batches: list[pd.DataFrame],
    reference_dists: list | None = None,
    n_processes: int = 0,
) -> pd.Series:
    """
    batches is list[pd.DataFrame], where each element is a batch of posterior probabilities.
    default number of processes is half the cores available. interpret n_processes < 1 as half of the available cores.
    """
    n_processes = os.cpu_count() // 2 if n_processes == 0 else n_processes  ## interpret
    reference_dists = (
        [np.mean(batch, axis=0) for batch in batches]
        if reference_dists is None
        else reference_dists
    )  # default to aggregate posterior
    assert len(batches) == len(reference_dists)

    if n_processes > 1:
        args = [
            (batch, reference_dist)
            for batch, reference_dist in zip(batches, reference_dists)
        ]
        with Pool(n_processes) as p:
            max_weight_predictions = p.starmap(
                max_weight_discretization, args, chunksize=1
            )
    else:
        max_weight_predictions = []
        for num, (batch, reference_dist) in enumerate(zip(batches, reference_dists)):
            max_weight_predictions.append(
                max_weight_discretization(batch, reference_distribution=reference_dist)
            )

    max_weight_predictions = [
        pd.Series(pred_batch, index=batch.index)
        for pred_batch, batch in zip(max_weight_predictions, batches)
    ]
    max_weight_predictions = pd.concat(max_weight_predictions)

    return max_weight_predictions


# endregion


### OTHER
def prelim_threshold(probs, thresholds, uncoded_val=None):
    """
    probs is pandas DataFrame of (n_samples, n_classes)
    thresholds is (n_classes) or a float (that broadcasts)
    """
    assert np.all(thresholds >= 0.5)
    if np.any(thresholds == 0.5):
        print("warning: threshold at exactly 0.5 may have ties")
    hits = np.any(np.array(probs) >= thresholds, axis=1)
    discretized = pd.Series(np.argmax(probs[hits], axis=1), index=(probs.index[hits]))
    if uncoded_val is not None:
        discretized = discretized.reindex(probs.index, fill_value=uncoded_val)
    return discretized


def approximate_ml(
    training_probs,
    training_labels,
    testing_probs,
    testing_labels=None,
    model=None,
    **kwargs,
):
    """
    you can just include the training data in the testing data if you want to use that.
    """
    if model == None:
        model = svm.LinearSVC(C=100, **kwargs)
    model = model.fit(training_probs, training_labels)
    print("training accuracy", model.score(training_probs, training_labels))
    if testing_labels is not None:
        print("testing accuracy", model.score(testing_probs, testing_labels))

    model_outputs = model.predict(testing_probs)
    return model_outputs
