# group_regularization.py
import torch
import transformers


def group_lasso(model: torch.nn.Module, pruning_method: str):
    '''
    L1 regularization of groups of weights in weight matrix to encourage sparsity \\
    row=True regularizes rows, row=False regularizes cols\\
    ignores bias, layernorm, embedding params
    '''
    reg, layer_count = 0.0, 0.0
    filter_out = ["bias", "LayerNorm", "embed"]
    group_dim=1
    if pruning_method == 'col':
        group_dim=0
    for name, param in model.named_parameters():
        if not any(key in name for key in filter_out): # only fc weight matrices
            # if args.pruning_metho == "group_lasso": # sum of L2 norms of rows of matrix
            norms = torch.norm(param, dim=group_dim)
            reg += torch.sum(norms) # / norms.numel() # /numel() reduces magnitude of the norm to a manageable size but it scales rows differently depending on layer size. Might want to remove this if bigger rows should have more weight b/c they cause more latency
            layer_count += 1.0
    return reg / layer_count


if __name__=="__main__":
    model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    print(group_lasso(model, True))
# for name, param in model.named_parameters():
#     filter_out = ["bias", "LayerNorm"]
#     if not any(key in name for key in filter_out): # only fully connected layers
#         print(name)
#         print(param.shape)
        