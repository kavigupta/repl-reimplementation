from abc import ABC, abstractmethod

from .state import State


class Policy(ABC):
    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def initial_program_set(self):
        pass

    @abstractmethod
    def __call__(self, states):
        pass

    def roll_forward(self, specs):
        states = [(self.initial_state(spec), None) for spec in specs]
        states_sequences = []
        while not all(s.done for s, _ in states):
            states = self.update_states(states)
            states_sequences.append(states)
        return list(zip(*states_sequences))

    def update_states(self, states):
        not_done = [state for state, _ in states if not state.done]
        outputs = self(not_done)
        output_idx = 0

        new_states = []
        for state, _ in states:
            if state.done:
                new_states.append((state, None))
            else:
                distro = outputs[output_idx]
                best_action = distro.argmax().item()
                new_states.append((state.transition(best_action), best_action))
                output_idx += 1
        return new_states

    def initial_state(self, spec):
        return State(self.initial_program_set, spec)
