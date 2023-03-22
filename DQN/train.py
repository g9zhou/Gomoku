from Gomoku import *
from dqn import *
from config import config
import logging

if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True

    env = Gomoku(10)

    exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # model = Linear(env, config)
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
