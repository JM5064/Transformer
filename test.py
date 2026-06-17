import torch
from torch.utils.data.dataloader import DataLoader
from tokenizers import Tokenizer

from model.bumblebee.bumblebee import Bumblebee
from train import validate
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE
from data.special_chars import EOS, UNK


def test_model(model, test_set, batch_size):
    # Create dataloader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    loss_func = CrossEntropyLoss(label_smoothing=0.0)

    print("Testing model...")
    metrics = validate(model, test_loader, loss_func)

    print("Cross-entropy loss:", metrics['average_val_loss'])
    print("Perplexity:", 2.71828183 ** metrics['average_val_loss'])


def greedypredict(model, input_seq, merge_pairs_json, vocab_json, max_iters=20, ignore_eos=False, temperature=0):
    # Get vocab
    tokenizer = Tokenizer.from_file("data/wikitext103/hf_data_json.json")

    # Encode input
    tokenized_input = tokenizer.encode(input_seq)
    context = tokenized_input.ids

    # Potentially in-context learning if input length > seq_len ?????

    # Convert to tensor
    context = torch.tensor(context)

    # # Predict for max number of iterations or until EOS
    encoded_output = []
    model.eval()
    with torch.no_grad():
        context = context.to(DEVICE)

        for i in range(max_iters):
            preds = model(context)

            if temperature == 0:
                # greedy
                best = preds[0].argmax(dim=1)
                best = best.squeeze().tolist()
                best_next = best[-1]
            else:
                # Apply softmax
                softmax = torch.softmax(preds / temperature, dim=2)

                # Draw a random token weighted by probability for the next token
                best_next = torch.multinomial(softmax[0][-1], num_samples=1).item()         

            if EOS in tokenizer.decode([best_next]) and not ignore_eos:
                break
            else:
                encoded_output.append(best_next)
                context = context.squeeze().tolist()
                context = context[1:] + [best_next]
                context = torch.tensor(context)
                context = context.to(DEVICE)

    output = tokenizer.decode(encoded_output)
    print("Input:", input_seq)
    print()
    print("Output:", output)


if __name__ == "__main__":
    # Modify model path for desired model
    MODEL_PATH = 'best.pt'

    encoded_text_json = 'data/wikitext103/encoded_text_test.json'
    merge_pairs_json = 'data/wikitext103/merge_pairs.json'
    vocab_json = 'data/wikitext103/vocab.json'

    # Load dataset
    wikitext2_test = WikiText2(
        encoded_text_json=encoded_text_json,
        vocab_json=vocab_json
    )

    # Define params
    VOCAB_SIZE = wikitext2_test.get_vocab_size()
    BATCH_SIZE = 32

    # Load model
    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    model = model.to(DEVICE)

    model_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['state_dict']
    model.load_state_dict(model_state_dict)

    # Uncomment to test model
    # test_model(model, wikitext2_test, BATCH_SIZE)

    text = "The Montreal Canadiens , officially Club de hockey Canadien and colloquially known as the Habs , "
    greedypredict(model, text, merge_pairs_json, vocab_json, max_iters=50, ignore_eos=False, temperature=0.175)
    