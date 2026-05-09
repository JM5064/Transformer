import re, collections
from special_chars import UNK, EOS
import time


def get_pairs(chars: list[str]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)

    for i in range(len(chars) - 1):
        pairs[chars[i], chars[i + 1]] += 1

    return pairs


def merge_pair(pair: tuple[str, str], chars: list[str], vocab: dict[str, int], pairs: dict[tuple[str, str], int]) -> list[str]:
    merged = "".join(pair)
    new_chars = []

    # Remove pair from pairs
    del pairs[pair]

    i = 0
    while i < len(chars):
        if i+1 < len(chars) and (chars[i], chars[i+1]) == pair:
            new_chars.append(merged)
            vocab[chars[i]] -= 1
            vocab[chars[i+1]] -= 1
            vocab[merged] += 1

            if vocab[chars[i]] == 0:
                del vocab[chars[i]]
            if vocab[chars[i+1]] == 0:
                del vocab[chars[i+1]]

            # Update previous and next pairs with merged pair
            if i-1 >= 0:
                prev_pair = (chars[i-1], chars[i])
                new_pair = (chars[i-1], merged)

                pairs[prev_pair] -= 1
                pairs[new_pair] += 1

                if pairs[prev_pair] == 0:
                    del pairs[prev_pair]
            if i + 2 < len(chars):
                prev_pair = (chars[i+1], chars[i+2])
                new_pair = (merged, chars[i+2])

                pairs[prev_pair] -= 1
                pairs[new_pair] += 1

                if pairs[prev_pair] == 0:
                    del pairs[prev_pair]

            i += 1
        else:
            new_chars.append(chars[i])
        i += 1

    return new_chars


def split_chars(text_arr: list[str]) -> list[str]:
    chars = []
    for t in text_arr:
        # Filter for unicode characters up to greek and coptic
        for c in t:
            if ord(c) <= 1023:
                chars.append(c)
            else:
                # Unknown character
                chars.append(UNK)

        # Append end of text character after each sample
        chars.append(EOS)
        # chars.extend(list(t))

    # add BOW, EOW markers
    # sym = ".,:@- !?;[]()+=&%$#/\"'"
    # if chars[0] not in sym:
    #     chars[0] = "_" + chars[0]
    # for i in range(1, len(chars) - 1):
    #     if chars[i] in sym:
    #         continue
    #     if chars[i - 1] == " ":
    #         chars[i] = "_" + chars[i]
    #     if chars[i + 1] == " ":
    #         chars[i] = chars[i] + "_"
    # if chars[-1] not in sym:
    #     chars[-1] = chars[-1] + "_"

    return chars


def get_vocab(chars: list[str]) -> dict[str, int]:
    vocab = collections.Counter(chars)

    return vocab


def save_vocab(vocab, vocab_file):
    with open(vocab_file, 'a') as file:
        file.write('token,count\n')
        for key, value in vocab.items():
            file.write(f'{key},{value}\n')


def bpe(text, num_merges):
    s = time.time()
    chars = split_chars(text)
    print("Splitting chars took", (time.time() - s) * 1000, "ms")
    vocab = get_vocab(chars)
    pairs = get_pairs(chars)

    for i in range(num_merges):
        s = time.time()

        best_pair = max(pairs, key=pairs.get)

        chars = merge_pair(best_pair, chars, vocab, pairs)
        
        print("Merge ", i, "took", (time.time() - s) * 1000, "ms")

    return chars, vocab
