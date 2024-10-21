import torch 

class Object(object):
    pass

class ReplayBuffer(): 
    def __init__(self, capacity=10000): 
        self.n = 0 
        self.capacity = capacity 
        self.x_storage = torch.tensor([]) 
        self.y_storage = torch.tensor([]) 
        pass 
    def __len__(self):
         return self.n 
    def add(self, x, y): 
        if self.n >= self.capacity: 
            ## discard earliest observation 
            self.x_storage = self.x_storage[1:] 
            self.y_storage = self.y_storage[1:] 
            self.n -= 1 ## TODO -1 won't cut it for torch.cat 
        pass 
        ## append to storage 
        self.x_storage = torch.cat([self.x_storage, x]) ## empty tensors just dissappear in torch.cat 
        self.y_storage = torch.cat([self.y_storage, y]) 
        self.n += 1 
        pass 
    def sample(self, batch_size=32, idx_list=None, device=torch.device('cpu')): 
        if idx_list is None: 
            idx_list = torch.randint(0, self.n, [batch_size]) 
            pass  
        out = Object() ## transitions 
        out.x = self.x_storage[idx_list].to(device) 
        out.y = self.y_storage[idx_list].to(device) 
        return out 
    def clear(self, n=None): 
        'clears first `n` transitions, or all if `n is None`'
        if n is None: 
            n = self.n 
            pass 
        self.x_storage = self.x_storage[n:] 
        self.y_storage = self.y_storage[n:] 
        self.n = len(self.x_storage) 
        pass 
    def save(self, path): 
        d = {
            'x': self.x_storage, 
            'y_state': self.y_storage
            } 
        torch.save(d, path) 
        pass 
    def load(self, path): 
        d = torch.load(path) 
        self.x_storage = torch.cat([self.x_storage, d['x']]) 
        self.y_storage = torch.cat([self.y_storage, d['y']]) 
        self.n += d['x'].shape[0] 
        if self.n > self.capacity: 
            self.clear(self.n - self.capacity) 
            pass 
        pass 
    pass 
