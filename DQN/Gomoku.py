import numpy as np
import torch
from scipy.signal import convolve2d

class Gomoku:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

        horizontal_kernel = np.ones([1,5])
        vertical_kernel = np.ones([5,1])
        pos_diag_kernel = np.eye(5, dtype=np.uint8)
        neg_diag_kernel = np.fliplr(pos_diag_kernel)
        self.kernels = [horizontal_kernel, vertical_kernel, pos_diag_kernel, neg_diag_kernel]

        self.kernel_combine = np.eye(9)
        self.kernel_combine += np.fliplr(self.kernel_combine)
        self.kernel_combine[4,:] = 1
        self.kernel_combine[:,4] = 1

    def __check__(self, player):
        close_stone = 0
        for kernel in self.kernels:
            if (convolve2d(self.board == player, kernel, mode="valid") == 5).any(): 
                return True
        return False
    
    def __store_history__(self, prev_state, action, reward, state, is_terminal):
        self.history["state"].append(prev_state)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["next_state"].append(state)
        self.history["is_terminal"].append(is_terminal)

    def reset(self):
        self.move = 0
        self.board = np.zeros((self.dim, self.dim), dtype=np.uint8)
        self.vacant_list = list(np.arange(self.dim ** 2))

        self.history = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "is_terminal": []
        }
        
        return self.board, 1

    def board_raw(self):
        return self.board

    def look(self, player):
        board = (self.board == player) + (self.board == 3-player)*-1
        return board.reshape(self.dim, self.dim, 1)

    def return_vacant(self):
        return self.vacant_list

    def place_stone(self, player, stone):
        # print(stone, type(stone))
        if type(stone) == np.int64 or type(stone) == int or type(stone) == torch.Tensor:
            if type(stone) == torch.Tensor:
                stone = stone.numpy()
            x = int(stone/self.dim)
            y = int(stone % self.dim)
        else:
            x, y = int(stone[0]), int(stone[1])
        board = self.board.copy()
        if self.board[x,y] == 0:
            self.board[x,y] = player
            self.vacant_list.remove(x*self.dim+y)
            self.move += 1

        reward, is_terminal = self.get_reward(player, board, x, y)
        # self.__store_history__(board, (x,y), reward, self.board, is_terminal)

        return self.move%2+1, reward, is_terminal
    
    def get_reward(self, player, board, x ,y):
        reward = 0
        is_terminal = False
        if self.__check__(player):
            reward += 100
            is_terminal = True
        elif board[x,y] != 0:
            reward += -150
        elif self.move == self.dim ** 2:
            reward += 0
            is_terminal = True
        else:
            reward += -5
            square = np.pad(self.board, pad_width=4)[x:x+9, y:y+9]
            reward_b = self.find_neighbors(square, player)
            reward_w = self.find_neighbors(square, 3-player)
            reward += max(reward_b, reward_w)
            # reward += convolve2d(square == player, self.kernel_combine, mode="valid").squeeze()


        return reward, is_terminal
            
    def state_shape(self):
        return self.dim
    
    def num_actions(self):
        # num_action = 0
        # for row in self.vacant_list:
        #     num_action += len(self.vacant_list[row])
        
        return self.dim ** 2
    
    def save(self, path, filename):
        np.savez(path+filename+".npz", 
                 state=self.history["state"], 
                 action=self.history["action"], 
                 reward=self.history["reward"], 
                 next_state=self.history["next_state"], 
                 is_terminal=self.history["is_terminal"])
        

    def find_neighbors(self, square, player):
        total = 0
        vecs = [square[4, :], square[:,4], np.diag(square), np.diag(np.fliplr(square))]

        for vec in vecs:
            center = int(len(vec)/2)
            l = center
            r = center
            num_zero = 0
            
            if player == 0:
                if vec[center-1] == 0:
                    l = center - 1
                if vec[center+1] == 0:
                    r = center + 1
                while (l >= 0 and vec[l] == 0): 
                    l -= 1
                while (r < len(vec) and vec[r] == 0):
                    r += 1
                return r-l-1
            
            if vec[center] != player:
                l -= 1
                r += 1
                
            while (l >= 0 and vec[l] != 3-player): 
                if vec[l] == 0:
                    num_zero += 1
                l -= 1
            while (r < len(vec) and vec[r] != 3-player):
                if vec[r] == 0:
                    num_zero += 1
                r += 1
        
            total += r-l-1-num_zero-(1-(player == vec[center]))
        
        return total - 3*(player == vec[center])

