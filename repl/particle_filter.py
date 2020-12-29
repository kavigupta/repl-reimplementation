import numpy as np
import torch


@attr.s
class GenerationPhase:
    # whether it can be considered a candidate, e.g., a full executable program
    is_candidate = attr.ib()
    # whether or not it can have more tokens added to it
    extendable = attr.ib()

    @classmethod
    def sequential_program_categorizer(cls, program):
        return cls(is_candidate=True, extendable=True)

    @classmethod
    def loop_program_categorizer(cls, program):
        # assumes 1 is the </s> token
        if program[-1] == 1:
            return cls(is_candidate=True, extendable=False)
        else:
            return cls(is_candidate=False, extendable=True)


def particle_filter(
    transition_model,
    observation_model,
    observations,
    prior,
    objective,
    *,
    categorizer=GenerationPhase.sequential_program_categorizer,
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

        phase_of: a function that takes in a sequence and categorizes it into a Phase

        prior: two lists [x_0] and [w] of elements and their weights.

        objective: a function that takes in an x and outputs a number. The particle
            seen with the greatest objective value will be returned.

    Returns
        the best particle, as judged by objective.
    """
    candidates = []

    x_vals, weights = prior
    weights = torch.tensor(weights)
    for obs in observations:
        candidates += [x for x in x_vals if categorizer(x).is_candidate]
        x_vals = [x for x in x_vals if categorizer(x).is_extendable]

        obs_weights = observation_model(x_vals, obs)
        if obs_weights is not None:
            obs_weights = obs_weights.cpu()
            sampler_weights = (weights * obs_weights).numpy()
        else:
            sampler_weights = weights.numpy().astype(np.float32)
        # print(obs_weights)
        # print([objective(x) for x in x_vals])
        sampler_weights /= np.sum(sampler_weights)
        indices = rng.choice(sampler_weights.size, p=sampler_weights, size=n_particles)
        x_vals = [x_vals[i] for i in indices]
        x_vals, weights = transition_model(x_vals, rng)
        weights = torch.tensor(weights)

    return sorted(candidates, key=objective)


def repl_particle_filter(
    policy,
    value,
    spec,
    max_steps=100,
    n_particles=100,
    objective=lambda state: state.is_goal,
    *,
    rng
):
    def transition_model(states, rng):
        with torch.no_grad():
            actions = policy(states).sample(rng)

        new_states, weights = [], []
        for state, action in zip(states, actions):
            new_states.append(state.transition(action))
            weights.append(1)
        return new_states, weights

    def observation_model(states, observation):
        with torch.no_grad():
            if value is None:
                return None
            return value(states)

    observations = [None] * max_steps

    prior = [policy.initial_state(spec)], [1]

    return particle_filter(
        transition_model=transition_model,
        observation_model=observation_model,
        observations=observations,
        prior=prior,
        objective=objective,
        rng=rng,
        n_particles=n_particles,
    )[0]
