import torch 

from replay_buffer import ReplayBuffer, Object 

class GradReplayBuffer(ReplayBuffer): ## no utilized inheritance, just guaranteeing the API contract 
    '''Samples gradients from an SSR Agent's existing replay buffer. 
    Cannot modify the underlying buffer and fails to do so quietly. 
    '''
    def __init__(self, ssr_agent): 
        if not isinstance(ssr_agent.replay_buffer, ReplayBuffer): 
            raise ValueError('Error: ssr_agent.replay_buffer must be of type ReplayBuffer') 
        self.ssr_agent = ssr_agent 
        pass 
    def __len__(self): 
        return len(self.ssr_agent.replay_buffer) 
    def add(self, *args, **kwargs): 
        'No op silently. Modifying the underlying Replay Buffer is not permitted.' 
        pass 
    def sample(self, batch_size=32, idx_list=None, device=torch.device('cpu')): 
        if idx_list is None: 
            idx_list = torch.randint(0, self.n, [batch_size]) 
            pass 
        out = Object() ## structured like RL transitions 
        out.x = None 
        out.y = torch.cat([self.ssr_agent.get_grad_vec(idx) for idx in idx_list], dim=1, device=device) ## TODO refactor so observations are rows 
        return out 
    def clear(self, *args, **kwargs): 
        'No op silently. Modifying the underlying Replay Buffer is not permitted.' 
        pass 
    def save(self, *args, **kwargs): 
        'No op silently. Save via the underlying Replay Buffer interface.'
        pass 
    def load(self, *args, **kwargs): 
        'No op silently. Modifying the underlying Replay Buffer is not permitted.'
        pass 
    pass 

