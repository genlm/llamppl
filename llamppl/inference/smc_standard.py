import asyncio
import copy
from datetime import datetime
import warnings
import numpy as np

from ..util import logsumexp
from .smc_record import SMCRecord


def stratified_resample(weights):
    # source: https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html#stratified_resample
    """Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.

    Args:
        weights (list-like of float): List of weights as floats.

    Returns:
        (ndarray): Array of indexes into the weights defining the resample.
            The index of the zeroth resample is indexes[0], etc.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


async def smc_standard(
    model,
    n_particles,
    ess_threshold=0.5,
    visualization_dir=None,
    json_file=None,
    resampling_method="multinomial",
):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
        model (llamppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.
        visualization_dir (str): Path to the directory where the visualization server is running.
        json_file (str): Path to the JSON file to save the record of the inference, relative to `visualization_dir` if provided.
        resampling_method (str): The method to use for resampling. Must be one of 'multinomial' or 'stratified'. Defaults to 'multinomial'.

    Returns:
        particles (list[llamppl.modeling.Model]): The completed particles after inference.
    """
    if not (0 <= ess_threshold <= 1):
        raise ValueError(
            f"Effective sample size threshold must be between 0 and 1. Got {ess_threshold}."
        )

    if resampling_method not in ["multinomial", "stratified"]:
        raise ValueError(
            f"Invalid resampling method: {resampling_method}. Must be one of 'multinomial' or 'stratified'."
        )

    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    await asyncio.gather(*[p.start() for p in particles])
    record = visualization_dir is not None or json_file is not None
    history = SMCRecord(n_particles) if record else None

    ancestor_indices = list(range(n_particles))
    did_resample = False
    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        # Record history
        if record:
            if len(history.history) == 0:
                history.add_init(particles)
            elif did_resample:
                history.add_resample(ancestor_indices, particles)
            else:
                history.add_smc_step(particles)

        # Normalize weights
        W = np.array([p.weight for p in particles])
        if np.all(W == -np.inf):
            # Avoid a nans in the normalized weights.
            # Could just terminate inference here, but keep running for now.
            did_resample = False
            continue
        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        # Resample if necessary
        if ess_threshold > 0 and (
            ess_threshold - logsumexp(normalized_weights * 2)
            < np.log(ess_threshold) + np.log(n_particles)
        ):
            probs = np.exp(normalized_weights)
            if resampling_method == "multinomial":
                # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
                ancestor_indices = [
                    np.random.choice(range(len(particles)), p=probs)
                    for _ in range(n_particles)
                ]
            elif resampling_method == "stratified":
                ancestor_indices = stratified_resample(probs)
            else:
                raise ValueError(f"Invalid resampling method: {resampling_method}.")

            if record:
                # Sort the ancestor indices
                ancestor_indices.sort()

            particles = [copy.deepcopy(particles[i]) for i in ancestor_indices]
            avg_weight = w_sum - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight

            did_resample = True
        else:
            did_resample = False

    if record:
        # Figure out path to save JSON.
        if visualization_dir is None:
            json_path = json_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_relative = (
                json_file
                if json_file is not None
                else f"{model.__class__.__name__}-{timestamp}.json"
            )
            json_path = f"{visualization_dir}/{json_file}"

        # Save JSON
        with open(json_path, "w") as f:
            f.write(history.to_json())

        # Web path is the part of the path after the html directory
        if visualization_dir is not None:
            print(f"Visualize at http://localhost:8000/smc.html?path={json_relative}")
        else:
            print(f"Saved record to {json_path}")

    return particles
