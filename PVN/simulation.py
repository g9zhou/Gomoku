from net import PVN
from monte_carlo import AlphaPlayer
import numpy as np
from Gomoku import *

weight = "result/result_20/best_policy.weights"
pvn = PVN(6,weight)

policy = AlphaPlayer(pvn.pvf, 5, 400, False)

def simulation():
    dim = 6
    i = np.random.randint(2)
    env = Gomoku(dim)
    if i % 2 == 0:
        players = ["human", policy]
    else:
        players = [policy, "human"]

    is_terminal = False
    turn = 0
    winner = -1
    while not is_terminal:
        player = players[turn%2]
        turn += 1
        print(env.curr_player())
        if player == "human":
            board = env.board
            board = (env.board == env.curr_player()) + -1*(env.board == 3-env.curr_player())
            print(board)
            stone = input('row and column please: ')
            x, y = stone.split(' ')
            action = np.int64(x)*dim+np.int64(y)
            print(action)
        else:
            action, _ = player.get_action(env)

        winner, is_terminal = env.place_stone(action)

    print("Player {} wins!".format(winner))

if __name__ == "__main__":
    simulation()