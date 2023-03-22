import numpy as np
import copy

def softmax(x):
    sigma = np.exp(x-np.max(x))
    sigma /= np.sum(sigma)
    return sigma


class TreeNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.Q = 0
        self.u = 0
        self.p = prior

    def expand(self, action_prior):
        for action, prob in action_prior:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, exp_c):
        action = max(self.children, key=lambda k: self.children[k].get_value(exp_c))
        return action, self.children[action] 
    
    def update(self, leaf_value):
        if self.parent:
            self.parent.update(-leaf_value)
        self.N += 1
        self.Q += 1.0*(leaf_value-self.Q) / self.N

    def get_value(self, exp_c):
        self.u = (exp_c*self.p*np.sqrt(np.log(self.parent.N)/(1+self.N)))
        return self.Q + self.u
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    

class MCTS:
    def __init__(self, pvf, exp_c=5, n_rollout=10000):
        self.root = TreeNode(None, 1.0)
        self.policy = pvf
        self.exp_c = exp_c
        self.n_rollout = n_rollout
    
    def __rollout__(self, env):
        node = self.root
        is_terminal = False
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.exp_c)
            _, is_terminal = env.place_stone(action)
        
        action_probs, _ = self.policy(env)

        if not is_terminal:
            node.expand(action_probs)
        
        leaf_value = self.evaluate_rollout(env)
        node.update(-leaf_value)


    def evaluate_rollout(self, env, limit=1000):
        player = env.curr_player()
        env_copy = copy.deepcopy(env)

        for _ in range(limit):
            winner, is_terminal = env_copy.get_status()
            if is_terminal:
                break
            vacant_list = env_copy.return_vacant()
            probs = np.random.rand(len(vacant_list))
            action = vacant_list[np.argmax(probs)]
            env_copy.place_stone(action)
        
        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, env):
        for _ in range(self.n_rollout):
            env_copy = copy.deepcopy(env)
            self.__rollout__(env_copy)
        return max(self.root.children, key=lambda k: self.root.children[k].N)

    def update_with_action(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    

class MCTSAlpha(MCTS):
    def __init__(self, pvf, exp_c=5, n_rollout=10000):
        super().__init__(pvf, exp_c, n_rollout)

    def __rollout__(self, env):
        node = self.root
        # print(self.root.children)
        is_terminal = False
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.exp_c)
            winner, is_terminal = env.place_stone(action)
        
        action_probs, leaf_value = self.policy(env)

        if not is_terminal:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0
            else:
                leaf_value = 1 if winner == env.curr_player() else -1

        node.update(-leaf_value)
        

    def get_move_probs(self, env, epsilon=1e-3):
        for _ in range(self.n_rollout):
            env_copy = copy.deepcopy(env)
            self.__rollout__(env_copy)

        act_visits = [(act, node.N) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/epsilon*np.log(np.array(visits) + 1e-10))

        return acts, act_probs


class RandomPlayer(MCTS):
    def __init__(self, pvf, exp_c=5, n_rollout=2000):
        super().__init__(pvf,exp_c,n_rollout)

    def reset_player(self):
        self.update_with_action(-1)

    def get_action(self, env):
        vacant = env.return_vacant()
        if len(vacant) > 0:
            action = self.get_move(env)
            self.update_with_action(-1)
            return action, 1

class AlphaPlayer(MCTSAlpha):
    def __init__(self, pvf, exp_c=5, n_rollout=2000, is_selfplay=False):
        super().__init__(pvf, exp_c, n_rollout)
        self.selfplay = is_selfplay

    def reset_player(self):
        self.update_with_action(-1)

    def get_action(self, env, epsilon=1e-3):
        vacant = env.return_vacant()
        if len(vacant) > 0:
            action_probs = np.zeros(env.action_dim)

            acts, probs = self.get_move_probs(env, epsilon)
            action_probs[list(acts)] = probs
            action = np.random.choice(acts, p=probs)
            if self.selfplay:
                self.update_with_action(action)
            else:
                self.update_with_action(-1)

            return action, action_probs
        
        