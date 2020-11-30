import torch 
import torch.nn as nn

from transformers import (
    BertConfig,
    # BertForSequenceClassification,
    BertTokenizer,
)
from modeling_profiled_bert import BertForSequenceClassification

import torchprof
# import torch.autograd.profiler as profiler
import cProfile
import pstats
from pstats import SortKey

import argparse
import os

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from parser import build_args

import pandas as pd
import time, datetime
import numpy as np

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def process_stats(filename):
    p = pstats.Stats(filename)
    p = p.strip_dirs()
    p.sort_stats(SortKey.CUMULATIVE) # sort by cum time, print top 10
    p.print_stats('modeling_profiled_bert', 10)

    # p.print_stats('forward',10) # only forward methods, top 10
    # p.print_callees('forward',10)

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        for i,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            if i >= args.n_trials:
                break
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # with profiler.profile(record_shapes=True) as prof:
                #     with profiler.record_function("model_inference"):
                # print('before  ',torch.cuda.memory_stats(args.device)['allocated_bytes.all.peak']/(2**20))
                outputs = model(**inputs)
    return 0
    
def evaluate_autograd_profiler(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        paths = [("BertForSequenceClassification", "bert", "encoder","layer","1"), ("BertForSequenceClassification", "bert", "encoder","layer","1","attention"),("BertForSequenceClassification", "bert", "encoder","layer","1","intermediate","dense"), ("BertForSequenceClassification", "bert", "encoder","layer","1","output","dense")]
        
        for i,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            if i >= args.n_trials:
                break
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # with profiler.profile(record_shapes=True) as prof:
                #     with profiler.record_function("model_inference"):
                # torch.cuda.synchronize()
                with torchprof.Profile(model, use_cuda=False, paths=paths) as prof:
                    outputs = model(**inputs)
        # print(prof.display(show_events=False))
        prof_str, prof_stats = prof.display(show_events=False)
        return prof_str, prof_stats
# getting "RuntimeError: Profiler is already enabled on this thread"  error and idk what to do. It happened after I added a torch.num_threads line. But removed line and it still sucks

def cprof_main():
    args=build_args()    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # setup for latency test:
    # args.device='cpu' # jsut want to test cpu for now
    args.model_name_or_path = 'bert-base-uncased'
    args.per_gpu_eval_batch_size=1
    argdict = vars(args)
    argdict['n_trials'] = 10
    # torch.set_num_threads(32)

    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

   
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        num_hidden_layers=2,
        # pruning_method=args.pruning_method,
        # mask_init=args.mask_init,
        # mask_scale=args.mask_scale,
    )
   

    model = model_class(
        # args.model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        # cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # print('hello0')
    model.to(args.device)
    evaluate(args, model, tokenizer)
    # torch.autograd.profiler.profile(enabled=False)
    # prof_str, prof_stats = evaluate_autograd_profiler(args,model,tokenizer)

    # make a dataframe with each column named according to cpu statistic
        # each row corresponds to dff dimension
        # fill rows for each dimension in for loop
    print(prof_stats)

    tested_dims=[]
    columns = []
    latencies = {} # want to reuse keys of prof_stats
    for key in prof_stats.keys():
        latencies[key]=[]
        columns.append(str(key))
    latencies['dff'] = []
   
    min_dff=768
    max_dff=768*4
    for dff in range(min_dff,min_dff+17,16):
        tested_dims.append(dff)

        config.intermediate_size = dff
        model = model_class(config=config)
        model.to(args.device)
        prof_str, prof_stats = evaluate_autograd_profiler(args,model,tokenizer)

        
        for key in prof_stats.keys(): # for each module examined, store the cpu time
            latencies[key].append(prof_stats[key].cpu_total/args.n_trials)
        latencies['dff'].append(dff)
    print(latencies)
    
    latency_dataframe = pd.DataFrame.from_dict(data=latencies, orient='columns')
    print(latency_dataframe)
    print(latency_dataframe[('BertForSequenceClassification', 'bert', 'encoder', 'layer', '1', 'intermediate', 'dense')])

    # prof = evaluate(args,model,tokenizer)
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

def latency_measurement():
    args=build_args()    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # args.device = 'cpu'

    # setup for latency test:
    # args.device='cpu' # jsut want to test cpu for now
    args.model_name_or_path = 'bert-base-uncased'
    args.per_gpu_eval_batch_size=1
    argdict = vars(args)
    argdict['n_trials'] = 100
    argdict['n_warmup'] = 10
    # torch.set_num_threads(8) # single threaded latencies are slower but have much lower std dev so better for testing (since xeon CPU is not realistic scenario anyways)
 
    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )

    # get batch
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        batch = next(iter(eval_dataloader))
        batch = tuple(t.to(args.device) for t in batch)

    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        num_hidden_layers=2,
    )
    
    print('Device: ', args.device)
    min_dff=768
    max_dff=768*8
    dims_to_test = range(min_dff,max_dff,64)
    # dims_to_test = [768,868]
    latencies = np.zeros((len(dims_to_test), args.n_trials), dtype=float)
    for i, dff in enumerate(tqdm(dims_to_test, desc="profiling")):
        config.intermediate_size = dff
        model = model_class(config=config)
        model.to(args.device)
        # warmup network to stabilize latencies
        for j in range(args.n_warmup):
            out = model(**inputs)
        for trial in range(args.n_trials):
            if args.device != 'cpu':
                torch.cuda.synchronize(args.device)
            then = time.time()
            out = model(**inputs)
            if args.device != 'cpu':
                torch.cuda.synchronize(args.device)
            now = time.time()
            latencies[i,trial] = now-then
    
    # process results
    dt = datetime.datetime.now()
    dt_string = dt.strftime("%m-%d-(%H:%M:%S)")
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__),'latency_viz', 'inference', dt_string))

    means = np.mean(latencies, axis=1)
    stds = np.std(latencies, axis=1)
    mins = np.min(latencies, axis=1)
    maxes = np.max(latencies, axis=1)
    # print('latencies ', latencies)moksha
    print('means ', means)
    print('stds ', stds)

    # print to tensorboard
    #TODO: So far latency scales with flops almost perfectly on cpu. Need to test on GPU to get more interesting latency model.. otherwise need to rethink my project. Mobile gpu would offer interesting scaling but not currently available
    for i in range(len(means)):
        writer.add_scalar('Mean Observed Latency', means[i], dims_to_test[i])
        writer.add_scalar('STD Observed Latency', stds[i], dims_to_test[i])
        writer.add_scalar('Min Observed Latency', mins[i], dims_to_test[i])
        writer.add_scalar('Max Observed Latency', maxes[i], dims_to_test[i])
    
    writer.close()
if __name__=='__main__':
    # cprof_main()
    latency_measurement()

    # STATSFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),'eval_stats.csv')
    # cProfile.run('main()', STATSFILE)
    # process_stats(STATSFILE)


    

# def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return self.bertseqforward(
#         input_ids,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None)
    
#     def bertseqforward(self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
