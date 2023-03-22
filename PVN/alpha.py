import numpy as np
from collections import defaultdict
from Gomoku import *
from monte_carlo import *
from net import PVN
from utils import *
import os

def random_pvf(env):
    vacant_list = env.return_vacant()
    prob = np.ones(len(vacant_list))
    return zip(vacant_list, prob/np.sum(prob)), 0


class Alpha():
    def __init__(self, env, model=None):
        self.env = env
        self.dim = env.dim

        self.lr = 1e-3
        self.epsilon = 1.0
        self.n_rollout = 400
        self.n_rollout_random = 1000
        self.exp_c = 5
        self.batch_size = 512
        self.replay_buffer = ReplayBuffer(10000)
        self.game_per_epoch = 1
        self.num_step = 5
        self.kl_bound = 0.02
        self.eval_freq = 50
        self.epoch = 1000
        self.best_win = 0.0

        self.random_playout = 1000
        self.pvn = PVN(self.dim, model)
        self.policy = AlphaPlayer(self.pvn.pvf, self.exp_c, self.n_rollout, True)

    def augement_data(self, data):
        augmented_data = []
        for state, prob, winner in data:
            for i in range(4):
                state_rot = np.array([np.rot90(s, i) for s in state])
                prob_rot = np.rot90(np.reshape(prob,(self.dim, self.dim)), i)
                augmented_data.append((torch.Tensor(state_rot[np.newaxis,:]), torch.Tensor(prob_rot.flatten()[np.newaxis,:]), torch.Tensor([winner])))

                state_flip = np.array([np.fliplr(s) for s in state_rot])
                prob_flip = np.fliplr(prob_rot)
                augmented_data.append((torch.Tensor(state_flip[np.newaxis,:]), torch.Tensor(prob_flip.flatten()[np.newaxis,:]), torch.Tensor([winner])))

        return augmented_data

    def collect_data(self, n_game):
        for _ in range(n_game):
            _, play_data = self.self_play(self.epsilon)
            augmented_data = self.augement_data(play_data)
            self.replay_buffer.add(augmented_data)


    def policy_update(self):
        state_batch, probs_batch, winner_batch = self.replay_buffer.sample(self.batch_size)
        old_probs, _ = self.pvn.pv(state_batch)
        for _ in range(self.num_step):
            loss = self.pvn.train_step(state_batch, probs_batch, winner_batch, self.lr)
            new_probs, _ = self.pvn.pv(state_batch)
            kl = np.mean(np.sum(old_probs*(np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)), axis=1))
            if kl > self.kl_bound * 4:
                break
        
        if kl > self.kl_bound * 2:
            self.lr /= 1.5
        
        return loss
    
    def policy_evaluate(self, n_games=10):
        alpha_player = AlphaPlayer(self.pvn.pvf, self.exp_c, self.n_rollout)
        random_player = RandomPlayer(random_pvf, self.exp_c, self.n_rollout_random)

        score = defaultdict(int)
        for i in range(n_games):
            print(i)
            winner = self.evaluate(alpha_player, random_player, i)
            score[winner] += 1
        win_ratio = score[1]/n_games
        print("win: {}, loss: {}, tie: {}".format(score[1], score[0], score[-1]))
        return win_ratio
    
    def self_play(self, epsilon):
        states, probs, players = [], [], []
        self.env.reset()
        while True:
            state = self.env.look()
            action, action_probs = self.policy.get_action(self.env, epsilon)
            player = self.env.curr_player()
            states.append(state)
            probs.append(action_probs)
            players.append(player)
            
            winner, is_terminal = self.env.place_stone(action)
            if is_terminal:
                players = np.array(players)
                if winner != -1:
                    players[players == winner] = 1
                    players[players != winner] == -1
                break
        return winner, zip(states, probs, players)
    
    def evaluate(self, player1, player2, i):
        if i % 2:
            players = [player1, player2]
        else:
            players = [player2, player1]

        self.env.reset()
        j = 0
        while True:
            player = players[j%2]
            action, _ = player.get_action(self.env)
            print(player, action)
            winner, is_terminal = self.env.place_stone(action)
            if is_terminal:
                break
            j += 1
        
        if winner == -1:
            return -1
        return 1 if player == player1 else 0


    def run(self):
        folder = 'result/result_{}'
        j = 0
        while (os.path.exists(folder.format(j))):
            j += 1
        folder = folder.format(j)
        os.makedirs(folder.format(j), exist_ok=True)
        print("Data will be saved to "+folder)
        loss_list = []
        win_ratio_list = []
        for i in range(self.epoch):
            print("collecting data ...")
            self.collect_data(self.game_per_epoch)
            info = "epoch: {}".format(i)
            print(self.replay_buffer.size)
            if self.replay_buffer.size > self.batch_size:
                print("update ...")
                loss = self.policy_update()
                info += ", loss: {}".format(loss)
                loss_list.append(loss)

            print(info)

            if (i+1)%self.eval_freq == 0:
                print("evaluate ...")
                win_ratio = self.policy_evaluate()
                win_ratio_list.append(win_ratio)
                self.pvn.save_model(os.path.join(folder, "policy.weights"))
                if win_ratio > self.best_win:
                    print("New best policy")
                    self.best_win= win_ratio
                    self.pvn.save_model(os.path.join(folder, "best_policy.weights"))

        filename = "results.npz"
        np.savez(os.path.join(folder, filename), loss=np.array(loss_list), win=np.array(win_ratio_list))
        
    
    
