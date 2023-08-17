import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

class GPT2WithActivationLogging(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.activations = []

    def forward(self, input_ids=None, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        self.activations = outputs.hidden_states
        return outputs

def load_model_and_tokenizer(model_path, config_path):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        config = GPT2Config.from_pretrained(config_path, output_hidden_states=True)
        model = GPT2WithActivationLogging.from_pretrained(model_path, config=config)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model and tokenizer: {str(e)}")
        return None, None

def process_sentence(model, tokenizer, sentence, model_name):
    try:
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
    except Exception as e:
        print(f"Failed to encode input: {str(e)}")
        return

    model(input_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for i, layer_activations in enumerate(model.activations):
        for j, token_activations in enumerate(layer_activations[0]):
            data = token_activations.detach().numpy()
            token_name = tokens[j]
            activation_values = " ".join([str(value) for value in data])
            print(f"Model: {model_name} - Layer {i} Token '{token_name}' Activations: {activation_values}")

# Update the path to where you've manually downloaded the model
model_path_gpt2xl = "/Users/migueldeguzman/Desktop/activation/gpt2-large/"
model_gpt2xl, tokenizer_gpt2xl = load_model_and_tokenizer(model_path_gpt2xl, model_path_gpt2xl)

while True:
    sentence_gpt2xl = input("Please input a sentence for GPT2-large (standard) model (type 'quit' to stop): ")
    if sentence_gpt2xl.lower() == 'quit':
        break
    process_sentence(model_gpt2xl, tokenizer_gpt2xl, sentence_gpt2xl, "GPT2-large")
