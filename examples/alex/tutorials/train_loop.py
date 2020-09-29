from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# picks tokenizer used for specified pretrained model
# next loads pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

import datasets
mrpc_sets = datasets.load_dataset('glue', 'mrpc')
print(mrpc_sets)

def encode(examples):
    return tokenizer.encode_plus(examples['sentence1'],examples['sentence2'], truncation=True, return_tensors="pt", padding="max_length", max_length=128)

train_data = encode(mrpc_sets['train'])
valid_data = encode(mrpc_sets['validation'])
# test_data = encode(mrpc_sets['test'])
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=16)
batch = next(iter(train_loader))
for batch,i in enumerate(train_loader):
    print('bloop')
trainer_args = TrainingArguments(
    output_dir='/home/ahoffman/research/transformers/examples/alex/tutorials/out',
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    # do_predict=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    fp16=False
)

from transformers.optimization import AdamW, get_constant_schedule_with_warmup
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=.1)
scheduler = get_constant_schedule_with_warmup(optimizer, 500)

trainer = Trainer(
    model=model,
    args=trainer_args, 
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=valid_data,
    optimizers=(optimizer,scheduler))

# loader = 
trainer.train()
