import json
from tokenizers import Tokenizer
import re
import time


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def save_to_file(data, file_path, indent=None):
    """Save data to json file

    Args:
        data (dict): object to save
        file_path (str): file name to save data to
        indent (int|str, optional): indent character/size for file, default None
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def remove_short_messages(member_data, min_length=5):
    """Removes short messages from a person's dataset"""
    return [message for message in member_data if len(message['Contents']) >= min_length]


def remove_long_messages(member_data, max_length=1000):
    """Removes long messages from a person's dataset"""
    return [message for message in member_data if len(message['Contents']) <= max_length]


def remove_messages_with_links(member_data):
    """Removes messages with urls from a person's dataset"""
    return [message for message in member_data if 
            ('http' not in message['Contents'] and '://' not in message['Contents'])]


def remove_quotes(member_data):
    """Removes messages that quote other people (ie. start with "> ")"""
    return [message for message in member_data if not message['Contents'].startswith("> ")]


def lowercase_member_data(member_data):
    member_data_copy = [message.copy() for message in member_data]
    for message in member_data_copy:
        message['Contents'] = message['Contents'].lower()
    
    return member_data_copy


def remove_duplicate_messages(member_data):
    """Removes duplicate messages from a person's dataset"""
    all_messages = set()
    member_data_copy = []

    for message in member_data:
        contents = message['Contents']

        if contents not in all_messages:
            all_messages.add(contents)
            member_data_copy.append(message)

    return member_data_copy


def remove_messages_in_common(members_data):
    """Removes messages that at least two people have in common from each person's dataset"""
    all_messages = {}

    # Add to all_messages dictionary
    # key: message, value: members who have said that message
    for i, member_data in enumerate(members_data):
        for message in member_data:
            contents = message['Contents']
            if contents not in all_messages:
                all_messages[contents] = set()

            all_messages[contents].add(i)

    common_messages = [contents for contents, members in all_messages.items() if len(members) > 1]
    common_messages = set(common_messages)

    refined_members_data = []
    for member_data in members_data:
        refined_members_data.append([message.copy() for message in member_data if message['Contents'] not in common_messages])

    return refined_members_data


def preprocess_contents(contents):
    rules = {
        r"([0-9]+),([0-9])+": r"\1 @,@ \2",         # 1,000         -> 1 @,@ 000
        r"([0-9]+)\.([0-9])+": r"\1 @.@ \2",        # 3.14          -> 3 @.@ 14

        r"([\S])-([\S])": r"\1 @-@ \2",             # hello-there   -> hello @-@ there

        r"([\S])(['’])([mst])" : r"\1 \2\3",        # I'm -> I 'm
        r"([\S])(['’])re" : r"\1 \2re",             # we're -> we 're
        r"([\S])(['’])ll" : r"\1 \2ll",             # we'll -> we 'll
        r"([\S])(['’])ve" : r"\1 \2ve",             # I've -> I 've

        r'(["“])([^"”]+)(["”])': r'\1 \2 \3',       # "hello there" -> " hello there "

        r"([\S])([\(\)\[\]\{\}])([\S])" : r"\1 \2 \3",  # ()[]{} -> ( ) [ ] { }

        r"([\S])([\.,!?:;]) ": r"\1 \2 ",           # hello, there  -> hello , there
        r"([\S\.,!?;])([\.,!?;])$": r"\1 \2 ",      # hello, -> hello , 
    }

    for rule, replacement in rules.items():
        contents = re.sub(rule, replacement, contents)

    return contents


def preprocess_member_data(member_data):
    member_data_copy = [message.copy() for message in member_data]

    for message in member_data_copy:
        contents_replaced = preprocess_contents(message['Contents'])
        
        message['Contents'] = contents_replaced

    return member_data_copy


def tokenize_data(member_data, tokenizer):
    member_data_copy = [message.copy() for message in member_data]

    for message in member_data_copy:
        message['Contents'] = tokenizer.encode(message['Contents']).ids

    return member_data_copy


def remove_messages_over_tokens(tokenized_member_data, max_tokens=128):
    return [message for message in tokenized_member_data if len(message['Contents']) <= max_tokens]


def pad_member_data(tokenizer_member_data, max_tokens=128, PAD_index=1):
    tokenized_member_data_copy = [message.copy() for message in tokenizer_member_data]

    for message in tokenized_member_data_copy:
        amount_to_pad = max_tokens - len(message['Contents'])
        message['Contents'].extend([PAD_index] * amount_to_pad)

    return tokenized_member_data_copy


def to_contents(member_data):
    return [message['Contents'] for message in member_data]


def count_messages_equaling_content(data, content):
    return len([message for message in data if content == message['Contents']])


if __name__ == "__main__":
    members = ['carman', 'dzss', 'friendlynoob', 'genzi', 'heibunny', 'jmoreojm', 'msn', 'rmj', 'tcray', 'willywonka']

    members_data = [load_json(f'data/uai_data/uai/{member}.json') for member in members]

    # Filter data
    members_data = [remove_short_messages(data) for data in members_data]
    # members_data = [remove_long_messages(data) for data in members_data]
    members_data = [remove_messages_with_links(data) for data in members_data]
    members_data = [remove_quotes(data) for data in members_data]
    members_data = [remove_duplicate_messages(data) for data in members_data]
    members_data = remove_messages_in_common(members_data)
    members_data = [preprocess_member_data(data) for data in members_data]

    # tokenizer = Tokenizer.from_file("data/wikitext103/hf_data_json.json")
    # members_data = [tokenize_data(data, tokenizer) for data in members_data]

    
    for member_data in members_data:
        print(len(member_data))


