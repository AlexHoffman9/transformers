import os
import time

import torch
import torch.nn as nn
from torch.optim import SGD

from modules.binarizer import L1ColBinarizer, L1RowBinarizer, MagnitudeBinarizer
from modules.masked_nn import MaskedLinear


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

class MLPNeuronMasked(nn.Module):
    def __init__(self,
        in_features: int,
        classes: int,
    ):
        super(MLPNeuronMasked, self).__init__()
        dim1=in_features
        dim2=int(in_features)
        dim3=int(in_features/4)
        self.lin1 = nn.Linear(in_features=dim1, out_features=dim2, bias=True)
        self.lin1.mask_scores = torch.ones(dim2, dtype=torch.float, requires_grad=True) # scores to regularize which will choose neurons
        self.lin1.mask = torch.ones(dim2, dtype=torch.float, requires_grad=False) # binary mask to disable neurons after pruning
        self.lin2 = nn.Linear(in_features=dim2, out_features=dim3, bias=True)
        self.lin2.mask_scores = torch.ones(dim3, dtype=torch.float, requires_grad=True)
        self.lin2.mask = torch.ones(dim2, dtype=torch.float, requires_grad=False)
        self.lin3 = nn.Linear(in_features=dim3, out_features=classes, bias=True)
        self.in_features = in_features
        self.classes = classes
    
    def forward(self, inputs):
        x = self.lin1(inputs)
        x = nn.functional.relu(x)
        xmasked = x * self.lin1.mask_scores # Data type got changed to float64???
        x = self.lin2(xmasked)
        x = nn.functional.relu(x)
        x = x * self.lin2.mask_scores #* self.mask2
        x = self.lin3(x)
        return x

    # def prune_neurons(self, self. ):


    # def self_test(self, threshold=0.6):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=.1)
    #     dummy_data=torch.rand((self.in_features)).unsqueeze(0)
    #     dummy_target=torch.rand((self.classes)).unsqueeze(0)
    #     for thresh in [1.0,.8,.6,.4,.2, .1, .05]:
    #         print('Threshold = ', thresh)
    #         for i in range (5):
    #             optimizer.zero_grad()
    #             out = self.forward(dummy_data, threshold=thresh)
    #             # print(out.size())
    #             # print(out)
    #             loss=nn.functional.mse_loss(out, dummy_target)
    #             print(loss.item())
    #             loss.backward()
    #             optimizer.step()
    #     optimizer.zero_grad()
    #     out = self.forward(dummy_data, threshold=0.9)
    #     # print(out.size())
    #     print(out)
    #     loss=nn.functional.cross_entropy(out, torch.Tensor([1]).long())
    #     loss.backward()
    #     optimizer.step()

# test pruning with simple MLP and dummy data
# dim=10
# classes=10
# model = MLP(dim, classes, "topK")
# model.self_test(threshold=0.6)

# test with mnist
import torchvision

def mnist_eval(model, test_loader, threshold, device):
    total=0; correct=0
    for idx, batch in enumerate(test_loader):
        batch_x = torch.flatten(batch[0], start_dim=1).to(device)
        batch_y = batch[1].to(device)
        out = model(batch_x, threshold)
        pred_idx = torch.argmax(out, dim=-1)
        correct += torch.sum(pred_idx == batch_y)
        total+=batch_x.shape[0]
        
    accuracy = correct*1.0/(total)
    print('threshold: ', threshold, 'acc: ', accuracy)



def mnist_neuron_pruning(num_fine_tune_epochs, lambdas, model_dir, model_dir_out):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    train_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    test_batch_size=32
    test_data = torchvision.datasets.MNIST('/home/ahoffman/research/transformers/examples/alex/struct-pruning/emmental', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    model = MLPNeuronMasked(28*28, 10).to(device)
    # load pretrained model if possible to save training time
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
        model = model.to(device)
        print('loaded pretrained weights')
    target_dims = {'lin1.mask_scores': 28*28-50}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for reg_lambda in lambdas:
        for epoch in range(num_fine_tune_epochs):
            print('training epoch ', epoch)
            cum_loss=0.0
            for idx, batch in enumerate(test_loader):
                batch_x = torch.flatten(batch[0], start_dim=1).to(device)
                batch_y = batch[1].to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                reg = l1_reg_neuron(model)
                loss = nn.functional.cross_entropy(out, batch_y) + reg_lambda * reg
                loss.backward()
                optimizer.step()
                cum_loss+=loss
            post_prefixes = {'lin1': 'lin2', 'lin2':'lin3'} # gotta be a better way than this....
            prune_neuron_weights(model, target_dims, post_prefixes)
            
        
    # save model
    torch.save(model.state_dict(), model_dir_out)

def l1_reg_neuron(model):
    reg=0.0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            reg += torch.sum(param) # l1 norm is sum of weights
    return reg

# should just feed a list of layer names accessible in model.state_dict() so i can more easily parse
def prune_neuron_weights(model, target_dims: dict, post_prefixes:dict):
    '''prune model to match target dims\\
        target dims is dict of neuron mask name, dimension\\
        post prefixes is a dict of layer name, next layer name. The prunable layers surrounding neuron mask
    '''
    for name, param in model.named_parameters(): # BUG: my mask scores are not in the named parameters section because i didn't register them. should store them somewhere they're accessible or register them as parameters
        if "mask_scores" in name:
            _, idx = param.sort()
            num_to_keep = target_dims[name]
            # get prev and next layers
            prefix = name[:len(name)-12]
            post_prefix = post_prefixes[prefix]
            prev_weight = model.state_dict()[prefix+'.weight'] # mask will correspond to input layer
            post_weight = model.state_dict()[post_prefix+'.weight']
            # for name param in model.parameters
                # if prefix of current name matches
            # incoming_weights = model.named_parameters()[1]
            incoming_weights = incoming_weights[idx[:num_to_keep], :] # prune rows of incoming weight matrix
            # remove corresponding weights

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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

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


if __name__=="__main__":
    # mnist_pruning_test(5, os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model.pt"), pruning_method="row")
    mnist_neuron_pruning(5, [1], os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model.pt"), os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model_pruned.pt"))











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
