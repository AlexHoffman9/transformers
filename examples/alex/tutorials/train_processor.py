# Fine tuning bert to a downstream task using a processor to convert data->examples->features
# Alex Hoffman 28 09 2020
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import os, sys

def main(args, task, evaluate=False):
    train_dataset = load_and_cache_examples(args, task, evaluate)
    # examples = 
    # features = 


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args['local_rank'] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args['data_dir'],
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args['model_name_or_path'].split("/"))).pop(),
            str(args['max_seq_length']),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args['overwrite_cache']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args['model_type'] in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args['max_seq_length'],
            label_list=label_list,
            output_mode=output_mode,
        )
        if args['local_rank'] in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args['local_rank'] == 0 and not evaluate:
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


if __name__ == "__main__":
    # execute only if run as a script
    data_dir = "/home/ahoffman/research/transformers/data/glue/MRPC"
    args={'model_name_or_path':'mrpc', 'max_seq_length':256, 'overwrite_cache':False, 'data_dir':data_dir, 'local_rank':0, 'modle_type':'bert'}
    main(args, 'mrpc') # doing mrpc first