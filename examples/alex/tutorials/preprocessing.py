from transformers import AutoTokenizer
import torch

# download tokenizer for pretrained bert-base
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
print(tokenizer.decode(encoded_input["input_ids"]))

batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)

# batch of sentences
batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(batch)

# pair of sentences for certain tasks
# note that inputs are concatencated and token_type_ids are different for first and second sentence tokens
batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
                             "And I should be encoded with the second sentence",
                             "And I go with the very last one"]
encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
print(encoded_inputs)