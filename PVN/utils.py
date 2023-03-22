from collections import namedtuple
import random
import torch

transition = namedtuple("transition", "state, probs, winner")


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.size = 0
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        for arg in args[0]:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(transition(*arg))
            else:
                self.buffer[self.location] = transition(*arg)

            # Increment the buffer location
            self.location = (self.location + 1) % self.buffer_size
            self.size  = min(self.size+1, self.buffer_size)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch_samples = transition(*zip(*samples))
        states = torch.cat(batch_samples.state)
        probs = torch.cat(batch_samples.probs)
        winner = torch.cat(batch_samples.winner)
        return states, probs, winner