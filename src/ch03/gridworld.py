import sys
from contextlib import closing
from io import StringIO

import numpy as np
from gymnasium.envs.toy_text import cliffwalking

# actions の定義
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(cliffwalking.CliffWalkingEnv):
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
            P[s][DOWN] = self._transition_prob(position, [1, 0])
            P[s][LEFT] = self._transition_prob(position, [0, -1])
            
        # 初期の state の分布は一様分布 (uniform)
        isd = np.ones(self.nS) / slef.nS
        
        # 動的なプログラミングのための環境モデルを公開する
        # モデルフリー学習アルゴリズムには使われないでしょう
        self.P = P
        super(GridworldEnv, self).__init__(self.nS, self.nA, P, isd)
        
    def _limit_coodinates(self, coord):
        
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        
        return coord
    
    def _transition_prob(self, current, delta):
        
        # もし stuck が最後の状態なら
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        
        if current_state == 0 or current_state == self.nS - 1:
            return [(1.0, current_state, 0, True)]
        
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        
        is_done = new_state == 0 or new_state == self.nS - 1
        
        return [(1.0, new_state, -1, is_done)]
    
    def render(self, mode="human"):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # 最後の state の出力
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "
                
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
                
            outfile.write(output)
        output.write("\n")
        
        # human のためには何も返り値を返す必要はない
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()