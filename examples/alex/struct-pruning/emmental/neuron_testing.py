import os
import time
import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import SGD


from modules.binarizer import L1ColBinarizer, L1RowBinarizer, MagnitudeBinarizer
from modules.masked_nn import MaskedLinear

# TODO: check regularization now that i realized that I wasn't using the train loader (doh!)
    # train with regularization and prune and compare to random pruning, also compare to no regularization
        # plot results and summarize for presentation
        # port to BERT if it's promising!



# # Returns mask of inputs matrix
# # mask = 0 in rows of inputs with smallest L1 norm
# def l1_percentage_mask(inputs:torch.Tensor, threshold: float):
#     mask = inputs.clone() # use clone for gradient prop
#     # calculate norms of each matrix
#     L1_mean = torch.mean(inputs.abs(), dim=1)
#     # sort
#     _, idx = L1_mean.sort(descending=True)
#     num_to_keep = int(threshold*L1_mean.numel())
#     mask[idx[:num_to_keep],:] = 1.0 # largest num_to_keep rows are kept by writing one to their mask
#     mask[idx[num_to_keep:],:] = 0.0
#     return mask



class MLPNeuronMasked(nn.Module):
    def __init__(self,
        dims:tuple,
    ):
        super(MLPNeuronMasked, self).__init__()
        dim1,dim2,dim3,classes = dims
        self.lin1 = nn.Linear(in_features=dim1, out_features=dim2, bias=True)
        self.lin1.mask_scores = Parameter(torch.ones(dim2, dtype=torch.float), requires_grad=True) # scores to regularize which will choose neurons
        self.lin1.mask = Parameter(torch.ones(dim2, dtype=torch.float, requires_grad=False), requires_grad=False) # binary mask to disable neurons after pruning. Could instead remove weights to reshape neuron layer
        self.lin2 = nn.Linear(in_features=dim2, out_features=dim3, bias=True)
        self.lin2.mask_scores = Parameter(torch.ones(dim3, dtype=torch.float), requires_grad=True) # change back to true after getting pretrained
        self.lin2.mask = Parameter(torch.ones(dim3, dtype=torch.float, requires_grad=False), requires_grad=False)
        self.lin3 = nn.Linear(in_features=dim3, out_features=classes, bias=True)
        self.in_features = dim1
        self.classes = classes
    
    def forward(self, inputs):
        x = self.lin1(inputs)
        x = nn.functional.relu(x)
        x = x * self.lin1.mask_scores * self.lin1.mask
        x = self.lin2(x)
        x = nn.functional.relu(x)
        x = x * self.lin2.mask_scores * self.lin2.mask
        x = self.lin3(x)
        return x   


# test with mnist
import torchvision

def mnist_eval(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total=0; correct=0
        for idx, batch in enumerate(test_loader):
            batch_x = torch.flatten(batch[0], start_dim=1).to(device)
            batch_y = batch[1].to(device)
            out = model(batch_x)
            pred_idx = torch.argmax(out, dim=-1)
            correct += torch.sum(pred_idx == batch_y)
            total+=batch_x.shape[0]
            
        accuracy = correct*1.0/(total)
        print('eval acc: ', accuracy.item())
        return accuracy.item()

def mnist_train(model, optimizer, train_loader, device, reg_lambda):
    model.train()
    cum_loss=0.0
    for idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        batch_x = torch.flatten(batch[0], start_dim=1).to(device) # flatten 2d image for fc network
        batch_y = batch[1].to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        l1_reg = l1_reg_neuron(model) # changed for pretrained model
        ce_loss = nn.functional.cross_entropy(out, batch_y)
        loss = ce_loss + reg_lambda * l1_reg
        loss.backward()
        optimizer.step()
        cum_loss+=loss
    return cum_loss

def mnist_neuron_pruning(num_fine_tune_epochs, lambdas, model_dir, model_dir_out, sparsity_sched):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    train_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    test_batch_size=128
    test_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    # for reg_lambda in lambdas:
    reg_lambda = lambdas[0]
    model = MLPNeuronMasked((28*28, 300, 100, 10)).to(device)
    # load pretrained model if possible to save training time
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir), strict=False) # saved model doesn't have mask scores
        model = model.to(device)
        print('loaded pretrained weights')
    target_dims = {'lin1.mask_scores': 28*28-50}
    sparsity=1.0
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    print('\nTraining with reg lambda=', reg_lambda)

    for sparsity in sparsity_sched: # prune to each sparsity and fine tune
        # learn mask scores for an epoch
        for name, param in model.named_parameters():
            if "mask_scores" in name: # learn mask scores, mask stays frozen
                param.requires_grad = True
            else:                # freeze weights
                param.requires_grad = False
        for epoch in range(1): # train scores for one epoch
            print('\nlearning scores, sparsity:', sparsity)
            mnist_eval(model, test_loader, device)
            cum_loss = mnist_train(model, optimizer, train_loader, device, reg_lambda)
            
            # check neuron scores to see if they are diverging a bit
            scores, idx = model.lin1.mask_scores.sort()
            # print("regularization training epoch ", epoch)
            # print('mask scores mean: ', scores.mean(), 'std: ', scores.std()) # shoudl be 1 and 0
            # print(scores[:10], scores[-10:])
            # check if weights are growing to compensate. they are...
            weight = model.lin1.weight.data.abs()
            # print('lin1 weight mean', weight.mean())
        # torch.save(model.state_dict(), model_dir) # added to save pretrained model

        # fine tune weights for an epoch
        for name, param in model.named_parameters():
            if "mask" in name: # freeze mask scores and mask
                param.requires_grad = False
            else:                # freeze weights
                param.requires_grad = True
        # prune less important neurons and fine tune
        post_prefixes = {'lin1': 'lin2', 'lin2':'lin3'} # gotta be a better way than this....
        prune_pct_neuron_weights(model, sparsity, post_prefixes, prune_random=False) # random pruning enabled to compare to my method
        print('acc after pruning to sparsity:', sparsity)
        mnist_eval(model, test_loader, device)
        for epoch in range(1):
            cum_loss = mnist_train(model, optimizer, train_loader, device, reg_lambda) # masks frozen so regularization has no effect
        print('fine-tuned acc after pruning to sparsity:', sparsity, 'and one epoch fine-tuning')
        mnist_eval(model, test_loader, device)
    
    print('\n\nDone pruning, now training to convergence')
    max_acc=0
    convergence=0
    while convergence<2:
        cum_loss = mnist_train(model, optimizer, train_loader, device, reg_lambda) # masks frozen so regularization has no effect
        acc = mnist_eval(model, test_loader, device)
        if acc <= max_acc+.002:
            convergence += 1
        else:
            convergence = 0
        max_acc = max(max_acc, acc)


        
           
    # save model
    torch.save(model.state_dict(), model_dir_out)

def l1_reg_neuron(model):
    reg=0.0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            reg += torch.sum(param.abs()) # l1 norm is sum of weights
    return reg


# currently debugging pruning algorithm to make sure sparsity is met, binary masks stay binary, and mask scores are learned properly. Want to see results eventually as well... and transfer to bert next
def prune_pct_neuron_weights(model: nn.Module, sparsity: torch.float, post_prefixes:dict, prune_random=False):
    '''
    prunes to targeted sparsity at each prunable layer\
    '''
    with torch.no_grad():
        for name, mask in model.named_parameters(): 
            if "mask_scores" in name:
                prefix = name[:len(name)-12] # name of layer being masked
                # print(prefix, 'mean: ', mask.mean().item())
                # print(prefix, 'std: ', mask.std().item())
                
                binary_mask = model.state_dict()[prefix+'.mask']
                num_to_keep = math.floor(len(binary_mask)*sparsity) # how many neurons to not prune
                # prune least important neurons. Prunes mask down to desired percentage
                if not prune_random:
                    
                    vals, sorted_idx = mask.sort(descending=True) # pruned weights are already set to 0
                    # print(vals[-5:].item(), vals[:5].item())
                    # idx_to_keep, _ = idx[:num_to_keep].sort() # return top % row indices, sorted 
                    idx_to_prune = sorted_idx[num_to_keep:]  
                else: # random pruning. Needs to check which have already been pruned
                    # find weights which aren't pruned yet
                    num_active = int(binary_mask.sum())
                    num_to_prune = num_active-num_to_keep
                    prunable_idx = [idx for idx in range(len(binary_mask)) if binary_mask[idx]==1.0]
                    idx_to_prune = random.sample(prunable_idx, num_to_prune)
                # Remove mask
                binary_mask[idx_to_prune] = 0.0
                mask[idx_to_prune] = 0.0
                model.load_state_dict({prefix+'.mask': binary_mask, name: mask}, strict=False)
                


def test_pruned_model(pruned_model_dir):
    device='cuda'
    model = MLPNeuronMasked((28*28, 28*28, 7*28, 10)).to(device)
    if os.path.exists(pruned_model_dir):
        model.load_state_dict(torch.load(pruned_model_dir), strict=False) # saved model doesn't have mask scores
        # model = model.to(device)
        print('loaded pretrained weights')
    inputs = torch.rand((1,28**2)).to(device)
    # outputs = model(inputs) # purely for debugging purposes to see if intermediate representations are truly sparse (we haven't removed weights yet)
    # sparsity appears legit
    test_batch_size=16
    test_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    with torch.no_grad():
        print('\nmasked model: ')
        # outputs = model(inputs)
        mnist_eval(model, test_loader, device)

        # remove mask, zero prev weights and compare
        pre_weight = model.lin1.weight
        pre_bias = model.lin1.bias
        mask = model.lin1.mask.data
        pruned_idx = [idx for idx in range(len(mask)) if mask[idx]==0.0] #(mask==0.0).nonzero(as_tuple=False)
        mask2 = model.lin2.mask.data
        pruned_idx2 = [idx for idx in range(len(mask2)) if mask2[idx]==0.0]
        # remove mask
        for name, param in model.named_parameters():
            if 'mask' in name and 'scores' not in name:
                param.fill_(1.0)
        # for neuron in range(len(mask)):
            # if mask[neuron] < 0.5: # not 1.0, therefore prune
        pre_weight[pruned_idx,:] = 0.0
        pre_bias[pruned_idx] = 0.0
                # post_weight[:,data] = 0.0
        print('\nZeroed weights')
        # outputs = model(inputs)
        # mnist_eval(model, test_loader, device)

        # prune post weights
        post_weight = model.lin2.weight
        post_weight[:,pruned_idx] = 0.0 # should not change result since inputs to these weights are zero
        print(model.lin2.weight[0,0:5])
        # mnist_eval(model, test_loader, device)


def train_model(model_dir:str):
    device='cuda'
    model = MLPNeuronMasked((28**2,300,100,10)).to(device)
    train_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=True, download=True,
                        transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                        ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_batch_size=16
    test_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad=False
    for epoch in tqdm(range(10)):
        # mnist_train(model_small, optimizer, train_loader, device, 0.0)
        # mnist_eval(model_small, test_loader, device)
        mnist_train(model, optimizer, train_loader, device, 0.0)
        mnist_eval(model, test_loader, device)
    torch.save(model.state_dict(), model_dir)
        


if __name__=="__main__":
    # test_pruned_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model_pruned_10perc.pt"))

    # mnist_pruning_test(5, os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model.pt"), pruning_method="row")
    
    # train_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model-300-100_pruned.pt"))

    mnist_neuron_pruning(6, [0.001], os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model-300-100.pt"),
     os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model-300-100_pruned.pt"), [0.9,0.7,.5,.3,.2,.1,.075,.05,.04,.03,.02])
    
    
    # mnist_neuron_pruning(6, [0.001], os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model.pt"),
    #  os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model_pruned.pt"), [0.9,0.8,0.7,.6,.5,.4,.3,.2,.1])











# print(num_to_keep)
                # print(prefix,mask[:20])
                # print(prefix,binary_mask[:20])
                # currently able to replace model tensor directly but not by name and state_dict, which i kinda need fam
                # should really test feasibility with just the mask and ignore loading weights for now
                # prune weights: not working because state_dict() expects tensor of the same size...
                # prefix = name[:len(name)-12] # name of layer being masked
                # post_prefix = post_prefixes[prefix] # name of following layer given in dict
                # prev_weight = model.state_dict()[prefix+'.weight'] # mask will correspond to input layer
                # post_weight = model.state_dict()[post_prefix+'.weight']
                # model.lin1.weight = Parameter(prev_weight[idx_to_keep,:])
                # # model.load_state_dict({prefix+'.weight': prev_weight[idx_to_keep,:], post_prefix+'.weight': post_weight[:,idx_to_keep]}, strict=False)
                # # model.state_dict()[prefix+'.weight'] = prev_weight[idx_to_keep,:] # slice rows of prev weight matrix
                # # model.state_dict()[post_prefix+'.weight'] = post_weight[:,idx_to_keep] # slice cols of following weight matrix

                # prune bias
                # if bias exists..
                # prune mask scores

class MLP(nn.Module):
    def __init__(self,
        in_features: int,
        classes: int,
        pruning_method: str = "row",
    ):
        super(MLP, self).__init__()
        dim1=in_features
        dim2=int(in_features)
        dim3=int(in_features/4)
        self.lin1 = MaskedLinear(in_features=dim1, out_features=dim2, bias=True, mask_init="constant", mask_scale=0.0, pruning_method=pruning_method)
        self.lin2 = MaskedLinear(in_features=dim2, out_features=dim3, bias=True, mask_init="constant", mask_scale=0.0, pruning_method=pruning_method)
        self.lin3 = nn.Linear(in_features=dim3, out_features=classes, bias=True)
        self.in_features = in_features
        self.classes = classes
        self.pruning_method = pruning_method
    
    def forward(self, inputs, threshold):
        x = self.lin1(inputs, threshold)
        x = nn.functional.relu(x)
        x = self.lin2(x, threshold)
        x = nn.functional.relu(x)
        x = self.lin3(x, threshold)
        return x
    
    def self_test(self, threshold=0.6):
        optimizer = torch.optim.SGD(self.parameters(), lr=.1)
        dummy_data=torch.rand((self.in_features)).unsqueeze(0)
        dummy_target=torch.rand((self.classes)).unsqueeze(0)
        for thresh in [1.0,.8,.6,.4,.2, .1, .05]:
            print('Threshold = ', thresh)
            for i in range (5):
                optimizer.zero_grad()
                out = self.forward(dummy_data, threshold=thresh)
                # print(out.size())
                # print(out)
                loss=nn.functional.mse_loss(out, dummy_target)
                print(loss.item())
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()
        out = self.forward(dummy_data, threshold=0.9)
        # print(out.size())
        print(out)
        loss=nn.functional.cross_entropy(out, torch.Tensor([1]).long())
        loss.backward()
        optimizer.step()

def mnist_pruning_test(num_epochs, model_dir, pruning_method="row"):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    train_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    test_batch_size=128
    test_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    model = MLP(28*28, 10, pruning_method).to(device)
    # load pretrained model if possible to save training time
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
        model = model.to(device)
    else: # train
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        threshold=1.0
        for threshold in [0.6]: #,.8,.6,.4,.2]:
            for epoch in range(num_epochs):
                print('training epoch ', epoch)
                cum_loss=0.0
                for idx, batch in enumerate(test_loader):
                    batch_x = torch.flatten(batch[0], start_dim=1).to(device)
                    batch_y = batch[1].to(device)
                    optimizer.zero_grad()
                    out = model(batch_x, threshold)
                    loss = nn.functional.cross_entropy(out, batch_y)
                    cum_loss+=loss
                    loss.backward()
                    optimizer.step()
        # save model
        torch.save(model.state_dict(), model_dir)






# step 1: init pretrained model with neurons
# step 2: add regularization to loss function
# step 3: fine-tune, view neurons
# step 4: prune weights using neuron scores
    # eval

    # mnist_eval(model, test_loader, 1.0, device)
    # model.lin1.pruning_method="row"
    # model.lin2.pruning_method="row"
    # model.lin3.pruning_method="row"
    # model.lin3.bias.data = torch.zeros(model.lin3.bias.shape).to(device)
    # mnist_eval(model, test_loader, 0.6, device)
    # model.lin1.pruning_method="col"
    # model.lin2.pruning_method="col"
    # model.lin3.pruning_method="col"
    # mnist_eval(model, test_loader, 0.6, device)
    # making the final pruning method 'row' reduces performance a ton, but the outputs are not zeroed
    # this is because it zeros one of the output neurons, making it impossible to learn features specific to that one. Fix this by actually pruning neurons. Not row after row or col after col


# def prune_neurons()


# def prune_neurons_using_rows(model, mag_threshold, kperc, method="row"):
#     weights = (model.lin1.data, model.lin2.data)
#     L1_dim=1
#     if method=="row":
#         L1_dim=0
#     for weight in weights:
#         L1_mean = torch.mean(weight.abs(), dim=L1_dim)
#         _, idx = L1_mean.sort(descending=True)
#         num_to_keep = int(threshold*L1_mean.numel())
#         mask[idx[:num_to_keep],:] = 1.0 # largest num_to_keep rows are kept by writing one to their mask
#         mask[idx[num_to_keep:],:] = 0.0
#         return mask


# threshold=0.6 # percent to keep
# dim=10
# dim2=dim+2
# weight = torch.rand((dim,dim2))
# weight.requires_grad=True
# x = torch.rand((dim))
# row_mask = L1RowBinarizer.apply(weight, threshold)
# col_mask = L1ColBinarizer.apply(weight, threshold)
# # print(row_mask)

# trg = torch.ones((dim2))
# optimizer = SGD([weight], lr=.1)
# for mask in [row_mask, col_mask]:
#     optimizer.zero_grad()
#     masked_weight = mask*weight
#     out = torch.mv(masked_weight.transpose(0,1), x) # matrix vector mult
#     loss = torch.nn.functional.l1_loss(out, trg)
#     loss.backward()
#     print(out)
#     print(weight.grad)

# optimizer.zero_grad()
# # masked_weight = mask*weight
# out = torch.mv(weight.transpose(0,1), x) # matrix vector mult
# loss = torch.nn.functional.l1_loss(out, trg)
# loss.backward()
# print(out)
# print(weight.grad)
# print('x')
# print(x)

# for i in range(10):
#     optimizer.zero_grad()
#     masked_weight = row_mask*weight
#     out = torch.mv(masked_weight.transpose(0,1), x) # matrix vector mult
#     loss1 = torch.nn.functional.l1_loss(out, trg, reduction='sum')
#     loss1.backward()
#     optimizer.step()
#     print(loss1)
# prune threshold 



dims=[64,512,2048]
def runtime_analysis(dims):
    for dim in dims:
        threshold=0.5
        inputs = torch.rand((dim,dim))

        then=time.time()
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        now=time.time()
        print(dim,"elapsed:",now-then)

        # fixed magnitude threshold, no sorting
        then=time.time()
        mask=inputs>threshold
        now=time.time()
        print(dim," no sort elapsed:",now-then)

        # L1 mean sorting
        then=time.time()
        mask= l1_percentage_mask(inputs, threshold)
        now=time.time()
        print(dim," l1 elapsed:",now-then)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs.to(device)
    dims=[64,512,2048]
    for dim in dims:
        threshold=0.5
        inputs = torch.rand((dim,dim))

        then=time.time()
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        now=time.time()
        print(dim,"elapsed:",now-then)

        # fixed magnitude threshold, no sorting
        then=time.time()
        mask=inputs>threshold
        now=time.time()
        print(dim," no sort elapsed:",now-then)

        # L1 mean sorting
        then=time.time()
        mask= L1RowBinarizer.apply(inputs, threshold)
        now=time.time()
        print(dim," l1 elapsed:",now-then)

        # conclusion: sorting the L1 means is faster than sorting every weight, but slower than applying simple
        # magnitude threshold
