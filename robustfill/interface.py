import shelve
import tqdm
import numpy as np

from .ROB import generate_FIO
from .ROBUT import ALL_BUTTS

TOKENS = sorted(str(t) for t in ALL_BUTTS)
TOKEN_TO_INDEX = {t: i for i, t in enumerate(TOKENS)}
interpret_back = {str(t): t for t in ALL_BUTTS}


def generate(n_io):
    p, ins, outs = generate_FIO(n_io)
    return [str(t) for t in p.flatten()], ins, outs


def create_dataset(path, n, min_eg, max_eg):
    with shelve.open(path, "c") as s:
        for i in tqdm.tqdm(range(n)):
            i = str(i)
            if i not in s:
                s[i] = generate(np.random.randint(min_eg, max_eg + 1))


def interpret(token, state):
    fn = interpret_back[token]
    try:
        return fn(state), False
    except:
        return state, True
