import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

class RobertaWithActivationLogging(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.activations = []

    def forward(self, input_ids=None, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        self.activations = outputs.hidden_states
        return outputs

def load_model_and_tokenizer(model_path, config_path):
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        config = RobertaConfig.from_pretrained(config_path, output_hidden_states=True)
        model = RobertaWithActivationLogging.from_pretrained(model_path, config=config)
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

model_path_roberta = "roberta-base"
model_roberta, tokenizer_roberta = load_model_and_tokenizer(model_path_roberta, model_path_roberta)

while True:
    sentence_roberta = input("Please input a sentence for RoBERTa-base model (type 'quit' to stop): ")
    if sentence_roberta.lower() == 'quit':
        break
    process_sentence(model_roberta, tokenizer_roberta, sentence_roberta, "RoBERTa-base")
