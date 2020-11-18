import numpy as np
import torch


def particle_filter(
    transition_model,
    observation_model,
    observations,
    prior,
    objective,
    *,
    rng,
    n_particles
):
    """
    Runs a particle filter.
        transition_model: function that takes in a list of states [x_t] and an rng and returns a
            list of [x_{t + 1}] sampled from each distribution, along with a list of weights [w_{t + 1}]

        observation_model: function that takes in a list of states [x_t] and an observation y_t and returns a
            list of weights [w_t]. Each of these weights corresponds to the given x_t

        observations: a list of observations [y_t] to be made

        prior: two lists [x_0] and [w] of elements and their weights.

        objective: a function that takes in an x and outputs a number. The particle
            seen with the greatest objective value will be returned.

    Returns
        the best particle, as judged by objective.
    """
    best_x = (-float("inf"), None)

    x_vals, weights = prior
    weights = torch.tensor(weights)
    for obs in observations:
        best_x = max([best_x] + [(objective(x), x) for x in x_vals], key=lambda x: x[0])

        obs_weights = observation_model(x_vals, obs).cpu()
        sampler_weights = (weights * obs_weights).numpy()
        sampler_weights /= np.sum(sampler_weights)
        indices = rng.choice(sampler_weights.size, p=sampler_weights, size=n_particles)
        x_vals = [x_vals[i] for i in indices]
        x_vals, weights = transition_model(x_vals, rng)
        weights = torch.tensor(weights)

    best_x = max([best_x] + [(objective(x), x) for x in x_vals], key=lambda x: x[0])
    return best_x[1]


def repl_particle_filter(policy, value, spec, max_steps=100, n_particles=100, *, rng):
    def transition_model(states, rng):
        with torch.no_grad():
            actions = policy(states).sample(rng)
            print(set(actions))

        new_states, weights = [], []
        for state, action in zip(states, actions):
            new_states.append(state.transition(action))
            weights.append(1)
        return new_states, weights

    def observation_model(states, observation):
        with torch.no_grad():
            return value(states)

    observations = [None] * max_steps


    def objective(state):
        return state.is_goal()
    prior = [policy.initial_state(spec)], [1]

    return particle_filter(
        transition_model=transition_model,
        observation_model=observation_model,
        observations=observations,
        prior=prior,
        objective=objective,
        rng=rng,
        n_particles=n_particles,
    )
