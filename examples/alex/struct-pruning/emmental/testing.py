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
        self.lin3 = MaskedLinear(in_features=dim3, out_features=classes, bias=True, mask_init="constant", mask_scale=0.0, pruning_method=pruning_method)
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


def mnist_pruning_test(num_epochs, model_dir, pruning_method="row"):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    print(model.lin3.weight.shape)
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

    # eval

    # mnist_eval(model, test_loader, 1.0, device)
    mnist_eval(model, test_loader, 0.6, device)
    model.lin1.pruning_method="col"
    model.lin2.pruning_method="col"
    model.lin3.pruning_method="col"
    mnist_eval(model, test_loader, 0.6, device)
    # making the final pruning method 'row' reduces performance a ton, but the outputs are not zeroed
    # this is because it zeros one of the output neurons, making it impossible to learn features specific to that one. Fix this by actually pruning neurons. Not row after row or col after col


if __name__=="__main__":
    mnist_pruning_test(5, os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_mlp_model.pt"), pruning_method="row")











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
