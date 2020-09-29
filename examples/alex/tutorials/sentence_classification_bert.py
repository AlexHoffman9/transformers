from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# picks tokenizer used for specified pretrained model
# next loads pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

# class labels
classes = ["not paraphrase", "is paraphrase"]

# inputs
sequence_0 = "The company HuggingFace is in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# positive example
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")

# negative example
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

# output and softmax
paraphrase_classification_logits = model(**paraphrase)[0] # tensor of logits batch x classes
not_paraphrase_classification_logits = model(**not_paraphrase)[0]
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
