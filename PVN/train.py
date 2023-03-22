from Gomoku import *
from alpha import *


if __name__ == '__main__':
    env = Gomoku(6)
    alpha = Alpha(env)
    alpha.run()