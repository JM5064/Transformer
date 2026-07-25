import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tokenizers import Tokenizer

from model.bumblebee.bumblebee import Bumblebee
from train import validate
from model.loss import CrossEntropyLoss
from data.uai_data.uai_dataset import UAIDataset
from data.uai_data.process_uai_data import preprocess_contents
from utils import DEVICE


members = ['carman', 'dzss', 'friendlynoob', 'genzi', 'heibunny', 'jmoreojm', 'msn', 'rmj', 'tcray', 'willywonka']
# members = ['dzss', 'friendlynoob', 'jmoreojm', 'msn']


def test_model(model, test_set, batch_size):
    # Create dataloader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    loss_func = CrossEntropyLoss(label_smoothing=0.0)

    print("Testing model...")
    metrics = validate(model, test_loader, loss_func, classification=True)

    print("Cross-entropy loss:", metrics['average_val_loss'])
    print("Accuracy:", metrics['accuracy'])


def predict(model, input_sentence, tokenizer, pad_index=1, max_tokens=128):
    preprocessed_sentence = preprocess_contents(input_sentence)

    # Tokenize text
    input_tokens = tokenizer.encode(preprocessed_sentence).ids

    # Pad text
    amount_to_pad = max_tokens - len(input_tokens)
    input_tokens.extend([pad_index] * amount_to_pad)

    input_tokens = torch.tensor(input_tokens)
    input_tokens = input_tokens.to(DEVICE)

    with torch.no_grad():
        logits = torch.softmax(model(input_tokens), dim=1)

    pred = int(logits.argmax(dim=1))
    
    print("Probabilities:\n", logits)
    print("Input:", input_sentence)
    print("Prediction:", members[pred])


if __name__ == "__main__":
    # Modify model path for desired model
    MODEL_PATH = 'runs/uai/uai.pt'

    encoded_text_json = 'data/wikitext103/encoded_text_test.json'
    merge_pairs_json = 'data/wikitext103/merge_pairs.json'
    vocab_json = 'data/wikitext103/vocab.json'

    # Load dataset
    test_dataset = UAIDataset(split_path='data/uai_data/uai_dataset/test')

    # Define params
    VOCAB_SIZE = 30000
    BATCH_SIZE = 32
    SEQ_LEN = 128
    NUM_CLASSES = 10

    # Load model
    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)

    # Replace last layer
    model.unembedding = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=d_model*SEQ_LEN, out_features=NUM_CLASSES)
    )

    model_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['state_dict']
    model.load_state_dict(model_state_dict)

    model = model.to(DEVICE)
    model.eval()

    # Uncomment to test model
    # test_model(model, test_dataset, BATCH_SIZE)

    tokenizer = Tokenizer.from_file("data/wikitext103/hf_data_json.json")

    text = "tho im guessing some of them were in the train set"
    predict(model, text, tokenizer)
