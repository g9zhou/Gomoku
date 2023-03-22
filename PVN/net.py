import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        self.conv1 = nn.Conv2d(2, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.p1 = nn.Conv2d(64, 4, 1)
        self.p2 = nn.Linear(4*(dim ** 2), dim ** 2)

        self.v1 = nn.Conv2d(64, 2, 1)
        self.v2 = nn.Linear(2*(dim ** 2), 32)
        self.v3 = nn.Linear(32, 1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_p = F.relu(self.p1(x))
        x_p = x_p.view(-1, 4*(self.dim ** 2))
        x_p = F.log_softmax(self.p2(x_p), dim=1)

        x_v = F.relu(self.v1(x))
        x_v = x_v.view(-1, 2*(self.dim ** 2))
        x_v = F.relu(self.v2(x_v))
        x_v = torch.tanh(self.v3(x_v))
        return x_p, x_v
    
class PVN:
    def __init__(self, dim, model=None):
        self.dim = dim
        self.decay = 1e-4

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pvn = Network(dim).to(self.device)
        self.optimizer = optim.Adam(self.pvn.parameters(), weight_decay=self.decay)

        if model:
            self.pvn.load_state_dict(torch.load(model))

    def pv(self, state_batch):
        state_batch = torch.Tensor(state_batch).to(self.device)
        log_probs, value = self.pvn(state_batch)
        probs = np.exp(log_probs.detach().numpy())
        return probs, value.detach().numpy()

    def pvf(self, env):
        vacant = env.return_vacant()
        state = env.look()
        state.reshape(-1,2,self.dim, self.dim)
        log_probs, value = self.pvn(torch.Tensor(state).to(self.device))
        probs = np.exp(log_probs.detach().numpy().flatten())
        probs = zip(vacant, probs[vacant])
        value = value.detach()[0][0]
        return probs, value

    def train_step(self, state_batch, mcts_probs, win_mask, lr):
        state_batch = state_batch.to(self.device)
        mcts_probs = mcts_probs.to(self.device)
        win_mask = win_mask.to(self.device)

        self.optimizer.zero_grad()

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        log_probs, value = self.pvn(state_batch)
        v_loss = F.mse_loss(value.view(-1), win_mask)
        p_loss = -torch.mean(torch.sum(mcts_probs*log_probs, 1))
        loss = v_loss + p_loss

        loss.backward()
        self.optimizer.step()
        return loss.detach()


    def save_model(self, path):
        torch.save(self.pvn.state_dict(), path)
