import sys
from contextlib import closing
from io import StringIO

import numpy as np
from gym.envs.toy_text import discrete

# actions の定義
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """_summary_

    
    Args:
        discrete (_type_): _description_
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self):
        self.shape = (4, 4)
        self.nS = np.prod(self.shape)
        self.nA = 4
        P = {}
        
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            P[s][UP] = self._transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._trainsition_prob(position, [0, 1])
            