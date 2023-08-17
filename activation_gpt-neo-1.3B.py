import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def forward_hook(module, input, output):
    activations.append(output)

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path, output_hidden_states=True)
    model.eval()
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
            activation_values = ' '.join(map(str, token_activations.tolist())) # Convert activations to a space-separated string
            print(f"Layer {layer_num} Token '{token_name}' Activations: {activation_values}")

model_path_gptneo = "/Users/migueldeguzman/Desktop/activation/gpt-neo-1.3B"  # Local path to the model
model_gptneo, tokenizer_gptneo = load_model_and_tokenizer(model_path_gptneo)

sentence_gptneo = input("Please input a sentence for GPT-Neo1.3B model: ")
process_sentence(model_gptneo, tokenizer_gptneo, sentence_gptneo)

