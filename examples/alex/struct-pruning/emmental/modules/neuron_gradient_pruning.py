# pruning functions for use with neuron masked bert
import torch
import torch.nn as nn
import math
import random
import time

# def get_layer_names(model, suffix):
#     for name


def update_neuron_gradient_scores(model, use_abs_value=True):
    '''updates scores for bert model linear layers. Sums gradients of all incoming weights of neuron'''
    with torch.no_grad():
        for name, mask_scores in model.named_parameters():
            if 'neuron_scores' in name:
                prefix = name[:len(name)-14] # name of parameter being masked, i.e. bert.encoder.layer.5.intermediate
                w = [p for n,p in model.named_parameters() if (prefix+'.dense.weight') in n] # best way to get parameter by name... state dict did not give grad info
                weight_param=w[0]
                # score = h*dL/dh = Wx * dL/dh = sum wi * xi dL/dh = sum( wi*dL/dwi)
                weighted_grads = weight_param * weight_param.grad
                if use_abs_value:
                    weighted_grads = torch.abs(weighted_grads)
                else:
                    weighted_grads = -weighted_grads # trying opposite ranking to see if better. it is
                mask_scores += torch.sum(weighted_grads,dim=1) # sum weights which feed each neuron to get neuron score

def update_neuron_gradient_scores_mask(model, use_abs_value=True):
    '''accumulates gradient through neuron mask layer'''
    with torch.no_grad():
        for name, neuron_mask in model.named_parameters():
            if "neuron_mask" in name:
                prefix = name[:len(name)-12] # name of parameter being masked, i.e. bert.encoder.layer.5.intermediate
                grads = neuron_mask.grad
                if use_abs_value:
                    grads = grads.abs()
                else:
                    grads = -grads
                model.state_dict()[prefix+'.neuron_scores'] += grads



def global_neuron_prune(model, sparsity: torch.float, prune_random=False, dims=(12,3072), latency_lookup=[0.1,0.1,0.4,0.4,0.8,0.8,0.4,0.4,0.1,0.1,0.1,0.1]):
    normed_lat_lookup = torch.Tensor(latency_lookup)/sum(latency_lookup)
    n_layers=dims[0]
    ffn_width=dims[1]
    global_scores = torch.Tensor(n_layers*ffn_width) # preallocate tensor for all scores
    idx=0
    # concatenate all scores
    with torch.no_grad():
        for name, neuron_scores in model.named_parameters():
            if "neuron_scores" in name:
                global_scores[idx*ffn_width:(idx+1)*ffn_width] = neuron_scores*normed_lat_lookup[idx] # weight importance according to latency impact of layer
                idx += 1
        # now have tensor of all neuron scores, time to sort
        sorted_scores, idxs = global_scores.sort(descending=True)
        num_to_keep = math.floor(len(global_scores)*sparsity)
        threshold = sorted_scores[num_to_keep-1]
        # prune each layer given threshold
        idx=0
        print('global pruning with latency weighted layers, sparsity:', sparsity)
        actual_sparsity=0.0
        for name, neuron_scores in model.named_parameters():
            if "neuron_scores" in name:
                prefix = name[:len(name)-14] # name of layer being masked                
                binary_mask = model.state_dict()[prefix+'.neuron_mask']
                pruned_idx = (neuron_scores*normed_lat_lookup[idx])<threshold
                binary_mask[pruned_idx] = 0.0
                neuron_scores[pruned_idx] = -1e6
                sprs=binary_mask.sum()/len(binary_mask)
                print('layer ',idx, ' sparsity = ', sprs)
                actual_sparsity+=sprs
                idx+=1
        actual_sparsity = actual_sparsity/12
        print('actual sparsity: ', actual_sparsity)

def pruning_score_from_latencies(latencies:torch.Tensor, dimensions):
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
    

# initial idea: score neurons, save latency scores for each layer, globally sort neuron_score + latency_score, then prune n neurons and update latency scores
    # problem: If on the steep curve of the latency plot, the n neurons pruned will likely all belong to that layer, even if the latency plot is now different for neurons after first one
    # solution: Only prune one neuron at a time, then immediately update the latency scores
        # slow brute force method: grab all neuron scores, latency scores again and prune one more neuron, repeat
        # slightly faster: update the global scores of neurons belonging to the last pruned layer (index into global scores array, add difference in latency score), then prune again
    # alternative method:
        # consider latency score independently from global score
        # use latency score as probability for sampling layer to prune, then prune neurons from that layer
        # in expectation, we prune more from the layers which provide more speedup (although not immediate, we could be sitting on cliff of latency plot and fail to drop down due to stoch behavior)
        # don't need to globally sort anymore, but now we have no idea of relative neuron importance between layers, so it isn't really global pruning...
        # general problem is that once we prune, the layerwise statistics (latency, global importance) change, so we need to update both (recalc latency, resort importance scores)
        # bad idea is to use score threshold instead of sparsity threshold, because then i don't need to do any global sorting. Downside is calibration of score threshold (more hparams)
def global_neuron_prune_iterative(model, sparsity, prune_random=False, dims=(12,3072), latencies=[1]*12, prune_n=1, latency_lambda=1.0):
    # neuron scores param names
    neuron_score_names = [p for p,l in model.named_parameters() if "neuron_scores" in p]
    binary_mask_names = [p for p,l in model.named_parameters() if "neuron_mask" in p]

    n_layers=dims[0]
    ffn_width=dims[1]
    total_neurons = n_layers*ffn_width
    global_scores = torch.Tensor(n_layers*ffn_width) # preallocate tensor for all scores
    # concatenate all scores
    with torch.no_grad():
        actual_sparsity=42.0 # dummy
        while actual_sparsity > sparsity:
            then=time.time()
            current_layer_widths, pruning_scores = torch.zeros(n_layers, dtype=int), torch.zeros(n_layers, dtype=torch.float)
            idx=0
            # prune n lowest neurons (n>1 will make this faster since sorting is slow). Could replace resorting with updating probabilities for a stochastic approach
            # for name, neuron_scores in model.named_parameters():
            #     if "neuron_scores" in name:
            #         current_width = int(torch.sum(neuron_scores != -1e6).item()) # masked neurons have score == -1e9, so this gives us current number of neurons
            #         pruning_score = pruning_score_from_latencies(latencies, dimensions=[current_width])
            #         global_scores[idx*ffn_width:(idx+1)*ffn_width] = neuron_scores - pruning_score # subtract latency improvement score from importance score
            #         current_layer_widths[idx] = current_width
            #         pruning_scores[idx] = pruning_score.item()
            #         idx += 1
            for name in neuron_score_names:
                neuron_scores = model.state_dict()[name]
                current_width = int(torch.sum(neuron_scores != -1e6).item()) # masked neurons have score == -1e9, so this gives us current number of neurons
                pruning_score = pruning_score_from_latencies(latencies, dimensions=[current_width])
                # normalize scores within each layer so they are globally comparable if one layer is biased to higher or lower scores
                # BUG can't set neuron scores to -1e6 anymore
                global_scores[idx*ffn_width:(idx+1)*ffn_width] = neuron_scores/torch.norm(neuron_scores) + latency_lambda*pruning_score # high pruning_score = higher improvement in latency
                current_layer_widths[idx] = current_width
                pruning_scores[idx] = pruning_score.item()
                idx += 1
            
            # sort global neuron scores
            sorted_scores, idxs = global_scores.sort(descending=True)
            num_to_keep = sum(current_layer_widths) - prune_n # number of active neurons - number to prune per iteration
            threshold = sorted_scores[num_to_keep-1]

            # finding layer, idx from global idx
            pruned_global_idx = idxs[num_to_keep:num_to_keep+prune_n]
            pruning_locations = [(pruned_idx//ffn_width, pruned_idx%ffn_width) for pruned_idx in pruned_global_idx]
            # prune specified locations
            for layer,idx in pruning_locations:
                neuron_scores = model.state_dict()[neuron_score_names[layer]]
                neuron_scores[idx] = -1e6
                binary_mask = model.state_dict()[binary_mask_names[layer]]
                binary_mask[idx] = 0.0
                current_layer_widths[layer] = int(binary_mask.sum()) # take actual sum of sparsity 


            # DEBUG: which layers did we prune from? What were the latency scores?
            # pruned_global_idx = idxs[num_to_keep:num_to_keep+prune_n]
            # pruned_layers = pruned_global_idx // ffn_width
            # print("Pruning latency scores: ", pruning_scores)
            # print("pruned layers: ", pruned_layers)

            #BUG set threshold assuming every score will be unique but ended up seeing multiple neurons at same score, causing them all to meet threshold and get pruned
            # prune each layer given threshold. Alternative to this would be to get indices of pruned neurons, translate to layer, idx and directly prune (better for small number of prune_n)
            # idx = 0
            # # print('global pruning with latency weighted layers, sparsity:', sparsity)
            # for name, neuron_scores in model.named_parameters():
            #     if "neuron_scores" in name:
            #         prefix = name[:len(name)-14] # name of layer being masked                
            #         binary_mask = model.state_dict()[prefix+'.neuron_mask']
            #         pruned_idx = (neuron_scores-pruning_scores[idx])<threshold # calculated pruning scores on first pass through the network
            #         binary_mask[pruned_idx] = 0.0
            #         neuron_scores[pruned_idx] = -1e6
            #         current_layer_widths[idx] = int(binary_mask.sum().item())
            #         idx+=1
            actual_sparsity = float(torch.sum(current_layer_widths))/total_neurons
            now = time.time()
            # print(now-then)
        print('pruned to ', actual_sparsity, ' sparsity')
        print('Layer widths: ', current_layer_widths)

def layerwise_neuron_prune(model: nn.Module, sparsity: torch.float, prune_random=False):
    '''
    prunes to targeted sparsity at each prunable layer
    '''
    with torch.no_grad():
        for name, mask_scores in model.named_parameters(): 
            if "neuron_scores" in name:
                prefix = name[:len(name)-14] # name of layer being masked                
                binary_mask = model.state_dict()[prefix+'.neuron_mask']
                num_to_keep = math.floor(len(binary_mask)*sparsity) # how many neurons to not prune
                # prune least important neurons. Prunes mask down to desired percentage
                if not prune_random:
                    vals, sorted_idx = mask_scores.sort(descending=True) # pruned weights are already set to 0
                    idx_to_prune = sorted_idx[num_to_keep:]  
                else: # random pruning. Needs to check which have already been pruned
                    # find weights which aren't pruned yet
                    num_active = int(binary_mask.sum())
                    num_to_prune = num_active-num_to_keep
                    prunable_idx = [idx for idx in range(len(binary_mask)) if binary_mask[idx]==1.0] # where() did not work here
                    idx_to_prune = random.sample(prunable_idx, num_to_prune)
                # Remove mask
                binary_mask[idx_to_prune] = 0.0
                
                # if '10' in name: # debugging 
                #     print(binary_mask[:12])
                #     print('actual sparsity layer ',prefix,': ',torch.sum(binary_mask/len(binary_mask)).item(), ' Expected:', sparsity) # verify that correct number is being pruned
                #     print(mask_scores)
                #     if sparsity!=1.0:
                #         print('pruned mean scores', torch.mean(mask_scores[idx_to_prune]).item())
                #         print('kept mean scores', torch.mean(mask_scores[sorted_idx[:num_to_keep]]).item())
                #     # model.load_state_dict({prefix+'.mask': binary_mask, name: mask_scores}, strict=False) 
                #     # w = [p for n,p in model.named_parameters() if (prefix+'.weight') in n]
                #     # weight_param=w[0]
                #     # print('mean\n',weight_param.mean(), 'std\n',weight_param.std())
                mask_scores[idx_to_prune] = -1e9 # for method with recoverable neurons, i would want to leave the scores. With signed gradient scores, need this set to -1e9

def layerwise_group_prune(model: nn.Module, sparsity: torch.float, prune_random=False, group='row'):
    '''
    prunes to targeted sparsity at each prunable layer, ranking by magniutude of row or column of weights
    '''
    with torch.no_grad():
        for name, mask_scores in model.named_parameters():
            if 'neuron_scores' in name:
                prefix = name[:len(name)-14] # name of layer being masked       
                binary_mask = model.state_dict()[prefix+'.neuron_mask']
                weight = model.state_dict()[prefix+'.dense.weight']
                num_to_keep = math.floor(len(binary_mask)*sparsity) # how many neurons to leave
                group_dim = 1 if group=='row' else 0
                if not prune_random:
                    _, sorted_idx = weight.norm(dim=group_dim).sort(descending=True) # sort by group dim 
                    idx_to_prune = sorted_idx[num_to_keep:]
                else: # random pruning. Needs to check which have already been pruned
                    # find weights which aren't pruned yet
                    num_active = int(binary_mask.sum())
                    num_to_prune = num_active-num_to_keep
                    prunable_idx = [idx for idx in range(len(binary_mask)) if binary_mask[idx]==1.0]
                    idx_to_prune = random.sample(prunable_idx, num_to_prune)
                # Remove mask
                binary_mask[idx_to_prune] = 0.0
                model.load_state_dict({prefix+'.mask': binary_mask}, strict=False)         
        
