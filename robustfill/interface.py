import shelve
import tqdm
import numpy as np

from .ROB import generate_FIO
from .ROBUT import ALL_BUTTS

TOKENS = [str(t) for t in ALL_BUTTS]
interpret_back = {str(t): t for t in ALL_BUTTS}


def generate(n_io):
    p, ins, outs = generate_FIO(n_io)
    return [str(t) for t in p.flatten()], ins, outs


def create_dataset(path, n, min_eg, max_eg):
    with shelve.open(path, "c") as s:
        for i in tqdm.trange(n):
            i = str(i)
            if i not in s:
                s[i] = generate(np.random.randint(min_eg, max_eg + 1))


def interpret(token, state):
    return interpret_back[token](state)
