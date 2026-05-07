import re, collections


def get_pairs(chars: list[str]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)

    for i in range(len(chars) - 1):
        pairs[chars[i], chars[i + 1]] += 1

    return pairs


def merge_pair(
    pair: tuple[str, str], pair_freq: int, chars: list[str], vocab: dict[str, int]
) -> list[str]:
    merged = "".join(pair)
    new_chars = []

    i = 0
    while i < len(chars):
      if i+1 < len(chars) and (chars[i], chars[i+1]) == pair:
        new_chars.append(merged)
        vocab[chars[i]] -= 1
        vocab[chars[i+1]] -= 1
        vocab[merged] += 1
        i += 1
      else:
        new_chars.append(chars[i])
      i += 1

    return new_chars


def split_chars(text: str) -> list[str]:
    chars = list(text)

    # add BOW, EOW markers
    sym = ".,:@- !?;[]()+=&%$#/\"'"
    if chars[0] not in sym:
        chars[0] = "_" + chars[0]
    for i in range(1, len(chars) - 1):
        if chars[i] in sym:
            continue
        if chars[i - 1] == " ":
            chars[i] = "_" + chars[i]
        if chars[i + 1] == " ":
            chars[i] = chars[i] + "_"
    if chars[-1] not in sym:
        chars[-1] = chars[-1] + "_"

    return chars


def get_vocab(chars: list[str]) -> dict[str, int]:
    vocab = collections.defaultdict(int)

    for char in chars:
        vocab[char] += 1

    return vocab


def bpe(text, num_merges):
    chars = split_chars(text)
    vocab = get_vocab(chars)
    for _ in range(num_merges):
        pairs = get_pairs(chars)
        best_pair = max(pairs, key=pairs.get)
        chars = merge_pair(best_pair, pairs[best_pair], chars, vocab)
    return chars, vocab
