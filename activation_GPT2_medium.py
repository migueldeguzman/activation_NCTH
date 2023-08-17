import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

# Defining the class that logs activation layers
class GPT2WithActivationLogging(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # Initialize an empty list to store the activations
        self.activations = []

    def forward(self, input_ids=None, **kwargs):
        # Forward pass through the model, capturing the outputs
        outputs = super().forward(input_ids, **kwargs)
        # Store the hidden states (activations)
        self.activations = outputs.hidden_states
        return outputs

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path, config_path):
    try:
        # Load the tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        config = GPT2Config.from_pretrained(config_path, output_hidden_states=True)
        model = GPT2WithActivationLogging.from_pretrained(model_path, config=config)
        # Set the model to evaluation mode
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model and tokenizer: {str(e)}")
        return None, None

# New version of the process_sentence function
def process_sentence(model, tokenizer, sentence, model_name):
    try:
        # Encode the input sentence
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
    except Exception as e:
        print(f"Failed to encode input: {str(e)}")
        return
    # Forward pass through the model
    model(input_ids)
    # Convert ids to tokens for readability
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Print the activations for each layer and each token
    for i, layer_activations in enumerate(model.activations):
        for j, token_activations in enumerate(layer_activations[0]):
            data = token_activations.detach().numpy()
            token_name = tokens[j]
            activation_values = " ".join([str(value) for value in data])
            print(f"Model: {model_name} - Layer {i} Token '{token_name}' Activations: {activation_values}")

# Define the path of the GPT-2 XL model and load the model and tokenizer
model_path_gpt2xl = "gpt2-medium"
model_gpt2xl, tokenizer_gpt2xl = load_model_and_tokenizer(model_path_gpt2xl, model_path_gpt2xl)

# The main loop with updated calls to process_sentence
while True:
    sentence_gpt2xl = input("Please input a sentence for GPT2-medium (standard) model (type 'quit' to stop): ")
    if sentence_gpt2xl.lower() == 'quit':
        break
    process_sentence(model_gpt2xl, tokenizer_gpt2xl, sentence_gpt2xl, "GPT2-medium")
