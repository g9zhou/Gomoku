import os
import numpy as np
import time
import sys
import torch
from collections import defaultdict, deque
from utils import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchsummary import summary

class Timer:
    def __init__(self, enabled=False) -> None:
        super().__init__()
        self.enabled = enabled
        self.category_sec_avg = defaultdict(
            lambda: [0.0, 0.0, 0]
        )  # A bucket of [total_secs, latest_start, num_calls]

    def start(self, category):
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[1] = time.perf_counter()
            stat[2] += 1

    def end(self, category):
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[0] += time.perf_counter() - stat[1]

    def print_stat(self):
        if self.enabled:
            print("Printing timer stats:")
            for key, val in self.category_sec_avg.items():
                if val[2] > 0:
                    print(
                        f":> category {key}, total {val[0]}, num {val[2]}, avg {val[0] / val[2]}"
                    )

    def reset_stat(self):
        if self.enabled:
            print("Reseting timer stats")
            for val in self.category_sec_avg.values():
                val[0], val[1], val[2] = 0.0, 0.0, 0


class QN(object):
    def __init__(self, env, config, logger=None):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.timer = Timer(False)

        self.build()

    def build(self):
        pass

    @property
    def policy(self):
        return lambda state: self.get_action(state)

    def save(self):
        pass

    def initialize(self):
        pass

    def get_best_action(self, state):
        raise NotImplementedError
    
    def get_action(self, state):
        if np.random.random() < self.config.soft_epsilon:
            dim = self.env.state_shape()
            vacant_list = self.env.vacant_list
            return torch.tensor([np.random.choice(vacant_list)])
        else:
            return self.get_best_action(state)[0]

    def init_averages(self):
        self.avg_reward = -50.0
        self.max_reward = -50.0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = -50.0

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))
        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def add_summary(self, latest_loss, latest_total_norm, t):
        pass
        
    def train(self, exp_schedule, lr_schedule, run_idx):
        replay_buffer = ReplayBuffer(self.config.buffer_size)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0
        scores_eval = []
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        while t < self.config.nsteps_train:
            total_reward = 0
            self.timer.start("env.reset")
            state, next_player = self.env.reset()
            state = self.env.look(next_player)
            prev_player = next_player

            state = torch.Tensor(state).unsqueeze(0).float()
            self.timer.end("env.reset")
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                
                self.timer.start("get_action")
                best_action, q_vals = self.get_best_action(state)
                action = exp_schedule.get_action(best_action)
                self.timer.end("get_action")

                max_q_values.append(max(q_vals))
                q_values += list(q_vals)

                self.timer.start("env.step")
                nex_player, reward, done = self.env.place_stone(next_player, action)
                new_state = self.env.look(prev_player)
                prev_player = next_player
                self.timer.end("env.step")

                self.timer.start("replay_buffer.store_effect")
                new_state = (torch.Tensor(new_state).unsqueeze(0).float())
                replay_buffer.add(state, new_state, torch.Tensor([action]).float(), torch.Tensor([[reward]]).float(), torch.Tensor([[done]]).float())
                state = (torch.Tensor(self.env.look(next_player)).unsqueeze(0).float())
                self.timer.end("replay_buffer.store_effect")

                self.timer.start("train_step")
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)
                self.timer.end("train_step")

                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and (t % self.config.learning_freq == 0)):
                    self.timer.start("logging")
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    self.add_summary(loss_eval, grad_eval, t)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(
                            t + 1,
                            exact=[
                                ("Loss", loss_eval),
                                ("Avg_R", self.avg_reward),
                                ("Max_R", np.max(rewards)),
                                ("eps", exp_schedule.epsilon),
                                ("Grads", grad_eval),
                                ("Max_Q", self.max_q),
                                ("lr", lr_schedule.epsilon),
                            ],
                            base=self.config.learning_start,
                        )
                    self.timer.end("logging")
                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, self.config.learning_start))
                    sys.stdout.flush()
                    prog.reset_start()

                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                last_eval = 0
                self.timer.start("eval")
                scores_eval += [self.evaluate()]
                self.timer.end("eval")
                self.timer.print_stat()
                self.timer.reset_stat()

            # if ((t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq)):
            #     self.logger.info("Recording...")
            #     last_record = 0
            #     self.timer.start("recording")
            #     self.record()
            #     self.timer.end("recording")


        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        with open(self.config.output_path + "scores_{}.pkl".format(run_idx), "wb") as f:
            pickle.dump(scores_eval, f)
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t , replay_buffer, lr):
        loss_eval, grad_eval = 0, 0
        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            self.timer.start("train_step/update_step")
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)
            self.timer.end("train_step/update_step")

        if t % self.config.target_update_freq == 0:
            self.timer.start("train_step/update_param")
            self.update_target_params()
            self.timer.end("train_step/update_param")

        if t % self.config.saving_freq == 0:
            self.timer.start("train_step/save")
            self.save()
            self.timer.end("train_step/save")

        return loss_eval, grad_eval
    
    def evaluate(self, env=None, num_episodes=None):
        
        # states = []
        # new_states = []
        # actions = []
        # outputs = []
        # dones = []

        if num_episodes is None:
            self.logger.info("Evaluating...")

        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        replay_buffer = ReplayBuffer(self.config.buffer_size)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state, next_player = env.reset()
            state = env.look(next_player)
            prev_player = next_player
            t = 0

            while True:
                action = self.get_action(state[None])
                next_player, reward, done = env.place_stone(next_player, action)
                new_state = env.look(prev_player)
                prev_player = next_player

                replay_buffer.add(state, new_state, int(action), reward, done)
                
                # states.append(state)
                # new_states.append(new_state)
                # actions.append(int(action))
                # outputs.append(reward)
                # dones.append(done)
                
                state = env.look(next_player)
                

                total_reward += reward
                if done:
                    break

                t += 1

            rewards.append(total_reward)
        # np.savez("output.npz", state = states, new_state = new_states, action = actions, reward = outputs, done = dones)
        # print("Finish saving")
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)
        
        return avg_reward
    
    def run(self, exp_schedule, lr_schedule, run_idx):
        self.initialize()

        # if self.config.record:
        #     self.record()

        self.train(exp_schedule, lr_schedule, run_idx)

        # if self.config.record:
        #     self.record()


class DQN(QN):
    def __init__(self, env, config, logger=None):
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")
        super().__init__(env, config, logger)
        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

    def initialize_models(self):
        raise NotImplementedError
    
    def get_q_values(self, state:torch.Tensor, network: str) -> torch.Tensor:
        raise NotImplementedError
    
    def update_target(self) -> None:
        raise NotImplementedError
    
    def calc_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def add_optimizer(self) -> Optimizer:
        raise NotImplementedError

    def build(self):
        self.initialize_models()
        if hasattr(self.config, "load_path"):
            print("Loading parameters from file:", self.config.load_path)
            load_path = Path(self.config.load_path)
            assert (load_path.is_file()), f"Provided load_path ({load_path}) does not exist"
            self.q_network.load_state_dict(torch.load(load_path, map_location="cpu"))
            print("Load successful!")
        else:
            print("Initializing parameters randomly")

            def init_weights(m):
                if hasattr(m, "weight"):
                    nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
                if hasattr(m, "bias"):
                    nn.init.zeros_(m.bias)

            self.q_network.apply(init_weights)
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.add_optimizer()

    def initialize(self):
        assert (self.q_network is not None and self.target_network is not None), "WARNING: Networks not initialized. Check initialize_models"
        self.update_target()


    def add_summary(self, latest_loss, latest_total_norm, t):
        self.summary_writer.add_scalar("loss", latest_loss, t)
        self.summary_writer.add_scalar("grad_norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg_Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max_Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std_Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg_Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max_Q", self.max_q, t)
        self.summary_writer.add_scalar("Std_Q", self.std_q, t)
        self.summary_writer.add_scalar("Eval_Reward", self.eval_reward, t)

    def save(self):
        torch.save(self.q_network.state_dict(), self.config.model_output)

    def get_best_action(self, state: torch.Tensor) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            s = torch.Tensor(state).float()
            action_values = (self.get_q_values(s, "q_network").squeeze().to("cpu").tolist())

        actions_sorted = np.argsort(action_values)
        dim = state.shape[1]
        idx = dim ** 2 - 1
        while state[0, int(actions_sorted[idx]/dim), int(actions_sorted[idx]%dim)] != 0:
            idx -= 1

        action = actions_sorted[idx]
        return action, action_values
    
    def update_step(self, t, replay_buffer, lr):
        self.timer.start("update_ste/replay_buffer.sample")

        s_batch, sp_batch, a_batch, r_batch, done_mask_batch = replay_buffer.sample(self.config.batch_size)
        self.timer.end("update_step/replay_buffer.sample")

        assert(self.q_network is not None and self.target_network is not None), "WARNING: Networks not initialized. Check initialize_models"
        assert(self.optimizer is not None), "WARNING: Optimizer not initialized. Check add_optimizer"

        self.timer.start("update_step/converting_tensors")
        done_mask_batch = done_mask_batch.bool()
        self.timer.end("update_step/converting_tensors")

        self.timer.start("update_step/zero_grad")
        self.optimizer.zero_grad()
        self.timer.end("update_step/zero_grad")

        self.timer.start("update_step/forward_pass_q")
        q_values = self.get_q_values(s_batch, "q_network")
        self.timer.end("update_step/forward_pass_q")

        self.timer.start("update_step/forward_pass_target")
        with torch.no_grad():
            target_q_values = self.get_q_values(sp_batch, "target_network")
        self.timer.end("update_step/forward_pass_target")

        self.timer.start("update_step/loss_calc")
        loss = self.calc_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        self.timer.end("update_step/loss_calc")
        self.timer.start("update_step/loss_backward")
        loss.backward()
        self.timer.end("update_step/loss_backward")

        if self.config.grad_clip:
            self.timer.start("update_step/grad_clip")
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.config.clip_val
            ).item()
            self.timer.end("update_step/grad_clip")
        else:
            total_norm = 0

        self.timer.start("update_step/optimizer")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.timer.end("update_step/optimizer")
        return loss.item(), total_norm
    
    def update_target_params(self):
        self.update_target()


class Linear(DQN):
    def initialize_models(self):
        state_shape = self.env.state_shape()
        num_actions = self.env.num_actions()

        input_size = self.config.state_history*state_shape*state_shape
        self.q_network = nn.Linear(input_size, num_actions)
        self.target_network = nn.Linear(input_size, num_actions)

    def get_q_values(self, state: torch.Tensor, network: str = "q_network"):
        out = self.q_network(state.flatten(start_dim=1)) if network == "q_network" else self.target_network(state.flatten(start_dim=1))
        return out

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(
        self, 
        q_values: torch.Tensor, 
        target_q_values: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        done_mask: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.config.gamma
        num_actions = self.env.num_actions()

        Q_samp = rewards + done_mask.bitwise_not()*gamma*target_q_values.max(dim=1)[0]
        loss = F.mse_loss(Q_samp, (q_values*F.one_hot(actions.to(torch.int64), num_classes=num_actions)).sum(dim=1))
        return loss
    
    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(self.q_network.parameters())


class NatureQN(Linear):
    def initialize_models(self):
        state_shape = self.env.state_shape()
        num_actions = self.env.num_actions()
        num_kernel = 8

        conv_size = (state_shape-3)*(state_shape-3)*num_kernel
        self.q_network = nn.Sequential(nn.Conv2d(self.config.state_history, num_kernel, 4, 1, 0), 
                            nn.Tanh(), 
                            nn.Flatten(start_dim=1),
                            nn.Linear(conv_size, 64),
                            nn.Tanh(),
                            nn.Linear(64, num_actions))
        self.target_network = nn.Sequential(nn.Conv2d(self.config.state_history, num_kernel, 4, 1, 0), 
                            nn.Tanh(), 
                            nn.Flatten(start_dim=1),
                            nn.Linear(conv_size, 64),
                            nn.Tanh(),
                            nn.Linear(64, num_actions))
        
    def get_q_values(self, state, network):
        state = state.transpose(1,3)
        out = self.q_network(state) if network == "q_network" else self.target_network(state)
        return out
    