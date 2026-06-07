import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model.bumblebee.bumblebee import Bumblebee
from train import validate
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE
import data.bpe
from data.special_chars import EOS, UNK


def greedypredict(model, input, max_iters=20):
    # Get vocab
    merge_pairs = data.bpe.load_from_file('data/wikitext2/merge_pairs.json')
    vocab = data.bpe.load_from_file('data/wikitext2/vocab.json')
    encoding, decoding = data.bpe.make_mapping(vocab)

    # Encode input
    tokenized_input = data.bpe.apply_merge_pairs(input, merge_pairs)
    encoded_input = []
    for t in tokenized_input:
        encoded_input.append(encoding.get(t, encoding[UNK]))

    # Potentially in-context learning if input length > seq_len ?????

    # Take seq_len tokens from end of input (crop/pad if needed)
    l = 128
    if len(encoded_input) > l:
        context = encoded_input[-l:]
    elif len(encoded_input) < l:
        context = [encoding[EOS]] * (l - len(encoded_input)) + encoded_input
    else:
        context = encoded_input

    # Convert to tensor
    context = torch.tensor(context)

    # Predict for max number of iterations or until EOS
    encoded_output = []
    model.eval()
    with torch.no_grad():
        context = context.to(DEVICE)

        for i in range(max_iters):
            preds = model(context)
            best = preds[0].argmax(dim=1)
            best = best.squeeze().tolist()
            best_next = best[-1]

            if EOS in decoding[best_next]:
                break
            else:
                encoded_output.append(best_next)
                context = context.squeeze().tolist()
                context = context[1:] + [best_next]
                context = torch.tensor(context)
                context = context.to(DEVICE)

    tokenized_output = [decoding[t] for t in encoded_output]
    output = "".join(tokenized_output)
    print("Input:", input)
    print("Output:", output)


if __name__ == "__main__":
    # Load dataset
    wikitext2_test = WikiText2(encoded_text_json='data/wikitext2/encoded_text_test.json')

    # # FOR TESTING
    # wikitext2_val = WikiText2(encoded_text_json='data/wikitext2/encoded_text_mini.json')

    # Define params
    VOCAB_SIZE = wikitext2_test.get_vocab_size()
    BATCH_SIZE = 32

    # Create dataloaders
    test_loader = DataLoader(wikitext2_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    
    loss_func = CrossEntropyLoss(vocab_size=VOCAB_SIZE, label_smoothing=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)

    model = model.to(DEVICE)

    # Modify model path for desired model
    MODEL_PATH = 'runs/every128,attentiondropout/best.pt'

    model_state_dict = torch.load(MODEL_PATH)['state_dict']
    model.load_state_dict(model_state_dict)

    print("Testing model...")
    metrics = validate(model, test_loader, loss_func)
    print(metrics)

    greedypredict(model, "when the", 30)
    