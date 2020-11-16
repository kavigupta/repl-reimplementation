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
        transition_model: function that takes in a list of states [x_t] and returns a
            list of [x_{t + 1}] along with probability weights. Effectively
            returns a probability distribution p(x_{t + 1} | x_t)
            for each of the input x_t.

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

        obs_weights = torch.tensor(observation_model(x_vals, obs))
        sampler_weights = (weights * obs_weights).numpy()
        indices = rng.choice(sampler_weights.size, p=sampler_weights, size=n_particles)
        x_vals = [x_vals[i] for i in indices]
        x_vals, weights = transition_model(x_vals)
        weights = torch.tensor(weights)

    best_x = max([best_x] + [(objective(x), x) for x in x_vals], key=lambda x: x[0])
    return best_x[1]


def repl_particle_filter(policy, value, spec, max_steps=100, n_particles=100, *, rng):
    def transition_model(states):
        new_states, weights = [], []
        for state, action_probs in zip(states, policy(states)):
            new_states += [
                state.transition(action) for action in range(len(action_probs))
            ]
            weights.extend(action_probs)
        return new_states, weights

    def observation_model(states, observation):
        return value(states)

    observations = [None] * max_steps

    prior = [policy.initial_state], [1]

    def objective(state):
        return state.is_goal()

    return particle_filter(
        transition_model=transition_model,
        observation_model=observation_model,
        observations=observations,
        prior=prior,
        objective=objective,
        rng=rng,
        n_particles=n_particles,
    )
