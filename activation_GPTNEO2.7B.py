import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def load_model_and_tokenizer():
    model_name_or_path = "/Users/migueldeguzman/Desktop/activation/gpt-neo-2.7B"
    model = GPTNeoForCausalLM.from_pretrained(model_name_or_path, output_hidden_states=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

def process_sentence(model, tokenizer, sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    hidden_states = outputs.hidden_states

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Iterate through all layers
    for layer_num, layer_activations in enumerate(hidden_states):
        # Iterate through all tokens in the layer
        for token_num, token_activations in enumerate(layer_activations[0]):
            token_name = tokens[token_num]
            activation_values = ' '.join(['{:.8f}'.format(value) for value in token_activations.tolist()]) # Convert activations to a space-separated string with 2 decimal places
            print(f"Layer {layer_num} Token '{token_name}' Activations: {activation_values}")

# Example usage
model, tokenizer = load_model_and_tokenizer()
sentence = input("Please enter the sentence you want to process: ")
process_sentence(model, tokenizer, sentence)
