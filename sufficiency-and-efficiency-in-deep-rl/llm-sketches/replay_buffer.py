import torch 

class Object(object):
    pass

class ReplayBuffer(): 
    def __init__(self, capacity=10000): 
        self.n = 0 
        self.capacity = capacity 
        self.state_storage = torch.tensor([]) 
        self.action_storage = torch.tensor([]) 
        self.reward_storage = torch.tensor([]) 
        self.next_state_storage = torch.tensor([]) 
        self.done_storage = torch.tensor([]) 
        pass 
    def __len__(self):
         return self.n 
    def add(self, state, action, reward, next_state, done): 
        if self.n >= self.capacity: 
            ## discard earliest observation 
            self.state_storage = self.state_storage[1:] 
            self.action_storage = self.action_storage[1:] 
            self.reward_storage = self.reward_storage[1:] 
            self.next_state_storage = self.next_state_storage[1:] 
            self.done_storage = self.done_storage[1:] 
            self.n -= 1 
        pass 
        ## append to storage 
        self.state_storage = torch.cat([self.state_storage, state]) ## empty tensors just dissappear in torch.cat 
        self.next_state_storage = torch.cat([self.next_state_storage, next_state]) 
        self.reward_storage = torch.cat([self.reward_storage, reward]) 
        self.action_storage = torch.cat([self.action_storage, action]) 
        self.done_storage = torch.cat([self.done_storage, done]) 
        self.n += 1 
        pass 
    def sample(self, batch_size=32, idx_list=None, device=torch.device('cpu')): 
        if idx_list is None: 
            idx_list = torch.randint(0, self.n, [batch_size]) 
            pass  
        out = Object() ## transitions 
        out.state = self.state_storage[idx_list].to(device) 
        out.next_state = self.next_state_storage[idx_list].to(device) 
        out.action = self.action_storage[idx_list].to(device) 
        out.reward = self.reward_storage[idx_list].to(device) 
        out.done = self.done_storage[idx_list].to(device) 
        return out 
    def clear(self, n=None): 
        'clears first `n` transitions, or all if `n is None`'
        if n is None: 
            n = self.n 
            pass 
        self.state_storage = self.state_storage[n:] 
        self.action_storage = self.action_storage[n:] 
        self.reward_storage = self.reward_storage[n:] 
        self.next_state_storage = self.next_state_storage[n:] 
        self.done_storage = self.done_storage[n:] 
        self.n = len(self.state_storage) 
        pass 
    def save(self, path): 
        d = {
            'state': self.state_storage, 
            'next_state': self.next_state_storage, 
            'action': self.action_storage, 
            'reward': self.reward_storage, 
            'done': self.done_storage 
            } 
        torch.save(d, path) 
        pass 
    def load(self, path): 
        d = torch.load(path) 
        self.state_storage = torch.cat([self.state_storage, d['state'].type(torch.int64)]) ## ints get loaded as floats, so casting 
        self.next_state_storage = torch.cat([self.next_state_storage, d['next_state'].type(torch.int64)]) 
        self.action_storage = torch.cat([self.action_storage, d['action']]) 
        self.reward_storage = torch.cat([self.reward_storage, d['reward']]) 
        self.done_storage = torch.cat([self.done_storage, d['done'].type(torch.int64)]) 
        self.n += d['done'].shape[0] 
        if self.n > self.capacity: 
            self.clear(self.n - self.capacity) 
            pass 
        pass 
    pass 
