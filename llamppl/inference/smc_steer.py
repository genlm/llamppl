import asyncio
import copy

import numpy as np

from ..util import logsumexp
from ..util import softmax


def find_c(weights, N):
    # Sort the weights
    sorted_weights = np.sort(weights)
    # Find the smallest chi
    B_val = 0.0
    A_val = len(weights)
    for i in range(len(sorted_weights)):
        chi = sorted_weights[i]
        # Calculate A_val -- number of weights larger than chi
        A_val -= 1
        # Update B_val -- add the sum of weights smaller than or equal to chi
        B_val += chi
        if B_val / chi + A_val - N <= 1e-12:
            return (N - A_val) / B_val
    return N


def resample_optimal(weights, N):
    c = find_c(weights, N)
    # Weights for which c * w >= 1 are deterministically resampled
    deterministic = np.where(c * weights >= 1)[0]
    # Weights for which c * w <= 1 are stochastically resampled
    stochastic = np.where(c * weights < 1)[0]
    # Stratified sampling to generate N-len(deterministic) indices
    # from the stochastic weights
    n_stochastic = len(stochastic)
    n_resample = N - len(deterministic)
    if n_resample == 0:
        return deterministic, np.array([], dtype=int), c
    K = np.sum(weights[stochastic]) / (n_resample)
    u = np.random.uniform(0, K)
    i = 0
    stoch_resampled = np.array([], dtype=int)
    while i < n_stochastic:
        u = u - weights[stochastic[i]]
        if u <= 0:
            # Add stochastic[i] to resampled indices
            stoch_resampled = np.append(stoch_resampled, stochastic[i])
            # Update u
            u = u + K
            i = i + 1
        else:
            i += 1
    # Concatenate the deterministic and stochastic resampled indices
    # resampled = np.concatenate((deterministic, stoch_resampled))
    # return resampled
    return deterministic, stoch_resampled, c


async def smc_steer(model, n_particles, n_beam):
    """
    Modified sequential Monte Carlo algorithm that uses without-replacement resampling,
    as described in [our workshop abstract](https://arxiv.org/abs/2306.03081).

    Args:
        model (llamppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to maintain.
        n_beam (int): Number of continuations to consider for each particle.

    Returns:
        particles (list[llamppl.modeling.Model]): The completed particles after inference.
    """
    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    await asyncio.gather(*[p.start() for p in particles])

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Count the number of finished particles
        n_finished = sum(map(lambda p: p.done_stepping(), particles))
        n_total = n_finished + (n_particles - n_finished) * n_beam

        # Create a super-list of particles that has n_beam copies of each
        super_particles = []
        for p in particles:
            p.untwist()
            super_particles.append(p)
            if p.done_stepping():
                p.weight += np.log(n_total) - np.log(n_particles)
            else:
                p.weight += np.log(n_total) - np.log(n_particles) - np.log(n_beam)
                super_particles.extend([copy.deepcopy(p) for _ in range(n_beam - 1)])

        # Step each super-particle
        await asyncio.gather(
            *[p.step() for p in super_particles if not p.done_stepping()]
        )

        # Use optimal resampling to resample
        W = np.array([p.weight for p in super_particles])
        W_tot = logsumexp(W)
        W_normalized = softmax(W)
        det_indices, stoch_indices, c = resample_optimal(W_normalized, n_particles)
        particles = [
            super_particles[i] for i in np.concatenate((det_indices, stoch_indices))
        ]
        # For deterministic particles: w = w * N/N'
        for i in det_indices:
            super_particles[i].weight += np.log(n_particles) - np.log(n_total)
        # For stochastic particles: w = 1/c * total       sum(stoch weights) / num_stoch = sum(stoch weights / total) / num_stoch * total * N/M
        for i in stoch_indices:
            super_particles[i].weight = (
                W_tot - np.log(c) + np.log(n_particles) - np.log(n_total)
            )

    # Return the particles
    return particles
