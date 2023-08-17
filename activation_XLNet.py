import torch
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

class XLNetWithActivationLogging(XLNetModel):
    def __init__(self, config):
        super().__init__(config)
        self.activations = []

    def forward(self, input_ids=None, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        self.activations = outputs.hidden_states
        return outputs

def load_model_and_tokenizer(model_path, config_path):
    try:
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        config = XLNetConfig.from_pretrained(config_path, output_hidden_states=True)
        model = XLNetWithActivationLogging.from_pretrained(model_path, config=config)
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

model_path_xlnet = "xlnet-base-cased"
model_xlnet, tokenizer_xlnet = load_model_and_tokenizer(model_path_xlnet, model_path_xlnet)

while True:
    sentence_xlnet = input("Please input a sentence for XLNet-base model (type 'quit' to stop): ")
    if sentence_xlnet.lower() == 'quit':
        break
    process_sentence(model_xlnet, tokenizer_xlnet, sentence_xlnet, "XLNet-base")
