from process_uai_data import *


def filter_members_data(members_data):
    members_data = [remove_short_messages(data) for data in members_data]
    members_data = [remove_messages_with_links(data) for data in members_data]
    members_data = [remove_quotes(data) for data in members_data]
    members_data = [remove_duplicate_messages(data) for data in members_data]
    members_data = remove_messages_in_common(members_data)

    # Speeds up filtering. Remove this if using an alternate splitting scheme:
    members_data = [member_data[:10000] for member_data in members_data]

    members_data = [preprocess_member_data(data) for data in members_data]

    return members_data


def tokenize_members_data(members_data, tokenizer):
    members_data = [tokenize_data(data, tokenizer) for data in members_data]

    members_data = [remove_messages_over_tokens(data) for data in members_data]
    members_data = [pad_member_data(data, PAD_index=1) for data in members_data]

    return members_data


def split_members_data(members_data, train_len=5000, val_len=1000, test_len=1000):
    members_data_train = [member_data[:train_len] for member_data in members_data]
    members_data_val = [member_data[train_len : train_len + val_len] for member_data in members_data]
    members_data_test = [member_data[train_len + val_len : train_len + val_len + test_len] for member_data in members_data]

    return members_data_train, members_data_val, members_data_test


if __name__ == "__main__":
    members = ['carman', 'dzss', 'friendlynoob', 'genzi', 'heibunny', 'jmoreojm', 'msn', 'rmj', 'tcray', 'willywonka']
    train_len = 5000
    val_len = test_len = 1000

    members_data = [load_json(f'data/uai_data/uai/{member}.json') for member in members]

    print("Filtering...")
    members_data = filter_members_data(members_data)
    
    print("Tokenizing...")
    tokenizer = Tokenizer.from_file("data/wikitext103/hf_data_json.json")
    members_data = tokenize_members_data(members_data, tokenizer)

    # Split dataset
    print("Splitting...")
    members_data_train, members_data_val, members_data_test = split_members_data(
        members_data, train_len, val_len, test_len
    )

    members_data_train = [to_contents(member_data) for member_data in members_data_train]
    members_data_val = [to_contents(member_data) for member_data in members_data_val]
    members_data_test = [to_contents(member_data) for member_data in members_data_test]

    print("Saving...")
    for i, member_data in enumerate(members_data_train):
        save_to_file(member_data, f'data/uai_data/uai_dataset/train/{members[i]}.json')

    for i, member_data in enumerate(members_data_val):
        save_to_file(member_data, f'data/uai_data/uai_dataset/val/{members[i]}.json')

    for i, member_data in enumerate(members_data_test):
        save_to_file(member_data, f'data/uai_data/uai_dataset/test/{members[i]}.json')


