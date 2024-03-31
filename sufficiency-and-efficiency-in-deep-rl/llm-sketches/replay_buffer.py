class Object(object):
    pass

class ReplayBuffer(): 
    def __init__(self, capacity=10000): 
        self.n = 0 
        self.capacity = capacity 
        self.state_list = [] 
        self.action_list = [] 
        self.reward_list = [] 
        self.next_state_list = [] 
        self.done_list = [] 
        pass 
    def __len__(self):
         return self.n 
    def add(self, state, action, reward, next_state, done): 
        if self.n >= self.capacity: 
            ## discard earliest observation 
            self.state_list = self.state_list[1:] 
            self.action_list = self.action_list[1:] 
            self.reward_list = self.reward_list[1:] 
            self.next_state_list = self.next_state_list[1:] 
            self.done_list = self.done_list[1:] 
            self.n -= 1 
        pass 
        ## cast to torch  
        state = torch.tensor(state) 
        action = torch.tensor(action) 
        reward = torch.tensor(reward) 
        next_state = torch.tensor(next_state) 
        done = torch.tensor(done) 
        ## append to buffer 
        self.state_list.append(state) 
        self.action_list.append(action) 
        self.reward_list.append(reward) 
        self.next_state_list.append(next_state) 
        self.done_list.append(done) 
        self.n += 1 
        pass 
    def sample(self, batch_size=32, idx_list=None): 
        ## sample lists 
        out = Object() ## transitions 
        out.state = [] 
        out.action = [] 
        out.reward = [] 
        out.next_state = [] 
        out.done = [] 
        if idx_list is not None: 
            batch_size = len(idx_list) 
        for i in range(batch_size): 
            if idx_list is None: 
                idx = random.randint(0, self.n-1) 
            else: 
                idx = idx_list[i]
                pass 
            out.state.append(self.state_list[idx]) 
            out.action.append(self.action_list[idx]) 
            out.reward.append(self.reward_list[idx]) 
            out.next_state.append(self.next_state_list[idx]) 
            out.done.append(self.done_list[idx]) 
            pass 
        ## stack  
        out.state = torch.stack(out.state) 
        out.action = torch.stack(out.action) 
        out.reward = torch.stack(out.reward).reshape([-1,1]) 
        out.next_state = torch.stack(out.next_state) 
        out.done = torch.stack(out.done).reshape([-1,1]) 
        return out 
    def clear(self, n=None): 
        'clears first `n` transitions, or all if `n is None`'
        if n is None: 
            n = self.n 
            pass 
        self.state_list = self.state_list[n:] 
        self.action_list = self.action_list[n:] 
        self.reward_list = self.reward_list[n:] 
        self.next_state_list = self.next_state_list[n:] 
        self.done_list = self.done_list[n:] 
        self.n = len(self.state_list) 
        pass 
    pass 
