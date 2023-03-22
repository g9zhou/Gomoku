import numpy as np
from scipy.signal import convolve2d

class Gomoku:
    def __init__(self, dim):
        self.dim = dim
        self.action_dim = dim ** 2
        self.reset()

        horizontal_kernel = np.ones([1,4])
        vertical_kernel = np.ones([4,1])
        pos_diag_kernel = np.eye(4, dtype=np.uint8)
        neg_diag_kernel = np.fliplr(pos_diag_kernel)
        self.kernels = [horizontal_kernel, vertical_kernel, pos_diag_kernel, neg_diag_kernel]

    def __check__(self, player):
        for kernel in self.kernels:
            if (convolve2d(self.board == player, kernel, mode="valid") == 4).any():
                return True
            
        return False
    
    def reset(self):
        self.move = 0
        self.board = np.zeros((self.dim, self.dim), dtype=np.uint8)
        self.vacant_list = list(np.arange(self.dim ** 2))
        self.is_terminal = False
        self.winner = -1
        self.last_move = -1


    def look(self):
        state = np.zeros((2, self.dim, self.dim), dtype=np.uint8)
        state[0, ...] = self.board == self.move % 2 + 1
        state[1, ...] = self.board == 2 - self.move % 2
        return state
    
    def return_vacant(self):
        return self.vacant_list
    
    def place_stone(self, stone):
        player = self.move % 2 + 1
        if type(stone) == np.int64:
            x = stone // self.dim
            y = stone % self.dim
        else:
            x, y = int(stone[0]), int(stone[1])
        self.board[x,y] = player
        self.vacant_list.remove(x*self.dim+y)

        winner, is_terminal = self.get_reward(player, x, y)
        self.move += 1
        self.is_terminal = is_terminal
        self.winner = winner
        self.last_move = x*self.dim+y
        return winner, is_terminal
    
    def get_status(self):
        return self.winner, self.is_terminal

    def get_reward(self, player, x, y):
        winner = 0
        is_terminal = False
        if self.__check__(player):
            winner = self.curr_player()
            is_terminal = True
        elif self.move == self.dim ** 2 - 1:
            winner = -1
            is_terminal = True
        
        return winner, is_terminal
    
    def curr_player(self):
        return self.move % 2 + 1
    
    
