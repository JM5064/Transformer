import torch
from torch.utils.data.dataloader import DataLoader

from model.bumblebee.bumblebee import Bumblebee
from train import validate
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE
import data.bpe
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
    merge_pairs = data.bpe.load_from_file(merge_pairs_json)
    vocab = data.bpe.load_from_file(vocab_json)
    encoding, decoding = data.bpe.make_mapping(vocab)

    # Encode input
    tokenized_input = data.bpe.apply_merge_pairs([input_seq], merge_pairs)[:-1]
    context = []
    for t in tokenized_input:
        context.append(encoding.get(t, encoding[UNK]))

    # Potentially in-context learning if input length > seq_len ?????

    # Convert to tensor
    context = torch.tensor(context)

    # Predict for max number of iterations or until EOS
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

            if EOS in decoding[best_next] and not ignore_eos:
                break
            else:
                encoded_output.append(best_next)
                context = context.squeeze().tolist()
                context = context[1:] + [best_next]
                context = torch.tensor(context)
                context = context.to(DEVICE)

    tokenized_output = [decoding[t] for t in encoded_output]
    output = "".join(tokenized_output)
    print("Input:", input_seq)
    print()
    print("Output:", output)


if __name__ == "__main__":
    # Modify model path for desired model
    MODEL_PATH = 'best103.pt'

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
    