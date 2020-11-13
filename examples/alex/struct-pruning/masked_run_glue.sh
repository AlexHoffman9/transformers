#!/bin/bash

SERIALIZATION_DIR=./glue_out
GLUE_DATA=/home/ahoffman/research/transformers/data/glue/MRPC
export CUDA_VISIBLE_DEVICES=0
# /data/ahoffman/anaconda3/envs/torch/bin/python /home/ahoffman/research/transformers/examples/alex/struct-pruning/masked_pruning_glue.py \
# --model_type bert --num_train_epochs 10 --per_gpu_train_batch_size 16 --initial_warmup 8 --max_seq_length 128 --learning_rate 2e-5 --tfwriter_dir_append sota_attempt

# neuron masked bert...does it run?
/data/ahoffman/anaconda3/envs/torch/bin/python /home/ahoffman/research/transformers/examples/alex/struct-pruning/masked_neuron_pruning_glue.py \
--model_type neuron_bert --num_train_epochs 4 --per_gpu_train_batch_size 8 --initial_warmup 8 --max_seq_length 128 --learning_rate 2e-5 --tfwriter_dir_append neuron_test \
--logging_steps 5

# See if pruning is affecting gradient prop
# /data/ahoffman/anaconda3/envs/torch/bin/python /home/ahoffman/research/transformers/examples/alex/struct-pruning/masked_pruning_glue.py \
# --final_threshold 0.9 --pruning_method magnitude --num_train_epochs 10  --initial_warmup 8
# for lambda in 0.0
# do
# /data/ahoffman/anaconda3/envs/torch/bin/python /home/ahoffman/research/transformers/examples/alex/struct-pruning/masked_pruning_glue.py \
# --final_threshold 0.5 --pruning_method row --regularization group_lasso --num_train_epochs 6 --final_lambda $lambda --initial_warmup 8
# done
# /data/ahoffman/anaconda3/envs/torch/bin/python /home/ahoffman/research/transformers/examples/alex/struct-pruning/masked_run_glue.py \
#     --output_dir $SERIALIZATION_DIR --overwrite_output_dir \
#     --data_dir $GLUE_DATA \
#     --do_train --do_eval --do_lower_case \
#     --model_type masked_bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name MRPC \
#     --per_gpu_train_batch_size 16 \
#     --warmup_steps 5400 \
#     --num_train_epochs 10 \
#     --learning_rate 3e-5 --mask_scores_learning_rate 1e-2 \
#     --initial_threshold 1 --final_threshold 0.15 \
#     --initial_warmup 1 --final_warmup 2 \
#     --pruning_method row --mask_init constant --mask_scale 0.