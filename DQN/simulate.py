import numpy as np
import torch.nn as nn

from Gomoku import *

WIN_REWARD = 100
LOSS_REWARD = -100
MOVE_REWARD = -10

np.random.seed(0)

def simulate(dim):
    board = Gomoku(dim)

    ishuman = False
    nextPlayer = 1

    while True:
        # state = board.look(nextPlayer)
        state = board.look(nextPlayer)
        print(state.squeeze(-1))
        
        x, y = policy(True)
        stone = (x, y)
        nextPlayer, reward, result = board.place_stone(nextPlayer, stone)
        print(reward)
        if result:
            print(board.look(nextPlayer).squeeze(-1))
            print("player {} lose!".format(nextPlayer))
            break

        input("\n press enter \n")
        
    


def policy(ishuman=True):
    if ishuman:
        player = input('row and column please: ')
        x, y = player.split(' ')
    else:
        x = np.random.choice(np.arange(dim))
        y = np.random.choice(np.arange(dim))
    
    return x, y

# def policy(state):
#     conv_size = (10-4)*(10-4)*4
#     q_network = nn.Sequential(nn.Conv2d(1, 4, 5, 1, 0), 
#                             nn.ReLU(), 
#                             nn.Flatten(start_dim=1),
#                             nn.Linear(conv_size, 64),
#                             nn.ReLU(),
#                             nn.Linear(128, 100))
#     path = "results/model.weights.pt"
#     q_network.load_state_dict(torch.load(path))
#     out = q_network(state)
#     idx = np.argmax(out)
#     return int(idx/10), int(idx%10)


if __name__ == '__main__':
    dim = 6
    simulate(dim)
