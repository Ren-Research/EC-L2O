import torch 
import torch.nn.functional as F


## Loss function
def object_loss_cost(demand, action, c = 0.1, debug = False):
    
    batch_size = demand.size(0)
    hit_cost = torch.norm(demand - action, dim=2)
    hit_cost = torch.norm(hit_cost, p='fro')**2
    
    #switch_diff = action[:,1:,:] - action[:,:-1,:]
    switch_cost = torch.norm(action[:,1:,:] - action[:,:-1,:], dim=2)
    switch_cost = torch.norm(switch_cost, p='fro')**2
    
    if c < 1e4:
        total_loss = 100*(hit_cost + c * switch_cost)
    else:
        total_loss = 100*switch_cost
        
    total_loss /= batch_size
    
    return total_loss

def object_loss_cr(demand, action, optimal_cost, min_cr = 2, c = 0.1, debug = False):
    
    # calculate the loss function based on the competitve ratio
    batch_size = demand.size(0)
    hit_cost = torch.norm(demand - action, dim=2)
    hit_cost = torch.norm(hit_cost, dim=1)**2
    
    #switch_diff = action[:,1:,:] - action[:,:-1,:]
    switch_cost = torch.norm(action[:,1:,:] - action[:,:-1,:], dim=2)
    switch_cost = torch.norm(switch_cost, dim=1)**2
    
    
    if c < 1e4:
        total_loss = (hit_cost + c * switch_cost)
    else:
        total_loss = switch_cost
    
    competitive_ratio = total_loss/optimal_cost
    
    
    competitive_ratio = F.relu(competitive_ratio - min_cr) + min_cr
    competitive_ratio = competitive_ratio.mean()
    
    return competitive_ratio

