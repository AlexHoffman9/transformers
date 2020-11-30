# code to generate a fake latency profile for use in testing my pruning methods
import torch
import math
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

# per dim linear ramp (slow) combined with steeper ramp every 'steps' steps

def noisy_linear_step(dim=768, steps=5, noise=.01, max_lat=1000, small_slope=.1, lpf=0):
    if small_slope<0 or small_slope>.9:
        print('Error: small slope must be between 0 and 0.9')
    overhead = max_lat*.1
    linear_range = max_lat*(0.9-small_slope)
    chunk_steps = math.floor(dim/steps) # trying to make noisy staircase
    latencies = torch.full(size=(dim,), fill_value=overhead)
    latencies += torch.arange(start=0.0, end=1.0, step=1/dim)*max_lat*small_slope # add small linear slope
    latencies += (torch.rand(dim) - .5)*linear_range*noise # noise
    for i in range(steps):
        min_idx = i * chunk_steps
        max_idx = min_idx + chunk_steps
        latencies[min_idx:max_idx] += i*linear_range/(steps-1) # add step to linear change
    if steps*chunk_steps < dim: # finish adding to final few idx
        min_idx = steps * chunk_steps
        max_idx = dim
        latencies[min_idx:max_idx] += (steps-1)*linear_range/(steps-1) # add step to linear change
    if lpf!=0:
        lpf_range = math.floor(lpf/2)
        lpf_latencies = latencies.clone().detach()
        # average internal latencies, leave latencies at edges alone
        for i in range(lpf_range,dim-lpf_range):
            lpf_latencies[i] = torch.mean(latencies[i-lpf_range:i+lpf_range+1])
        return lpf_latencies
    return latencies/latencies.mean()

# really basic flops approximator
def flops_linear(fc_dims=(768,768), pruned_dim_idx=0):
    max_flops = fc_dims[0]*fc_dims[1]
    flops = torch.arange(start=0, end=max_flops, step=fc_dims[1-pruned_dim_idx], dtype=torch.float)
    return flops/flops.mean()


def neuron_score_from_latencies(latencies:torch.Tensor, dimensions: torch.tensor):
    '''Outputs expected latency improvement score for pruning each layer\\
        Inputs: latencies dimension tensor (max_dim), dimensions tensor (n_layers)'''
    # assuming latency table is already smoothed...We can just subtract pruned latency from current latency. Bigger difference = bigger score
    scores = torch.zeros(len(dimensions))
    for layer,dim in enumerate(dimensions):
        if dim>=2:
            improvement = latencies[dim-1] - latencies[dim-2] # latency improvement, expected to be positive, so add to neuron scores
        else:
            improvement = 0
        scores[layer] = improvement
    if len(dimensions)==1:
        return scores[0]
    return scores

def selftest():
    dt = datetime.datetime.now()
    dt_string = dt.strftime("%m-%d-(%H:%M:%S)")
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__),'latency_viz',dt_string))
    latencies = noisy_linear_step(noise=0.0, lpf=11)
    # latencies = flops_linear()
    for i in range(len(latencies)):
        writer.add_scalar('lat', latencies[i], global_step=i)
    dims=torch.tensor(range(768), dtype=int)
    scores = neuron_score_from_latencies(latencies, dims)
    for i in range(len(dims)):
        writer.add_scalar('pruning_score', scores[i], global_step=dims[i])


if __name__ == '__main__':
    selftest()