import json
import time


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def remove_short_messages(member_data, min_length=5):
    """Removes short messages from a person's dataset"""
    return [message for message in member_data if len(message['Contents']) >= min_length]


def remove_long_messages(member_data, max_length=1000):
    """Removes long messages from a person's dataset"""
    return [message for message in member_data if len(message['Contents']) <= max_length]


def remove_contents_with_link(member_data):
    """Removes messages with urls from a person's dataset"""
    return [message for message in member_data if 
            ('http' not in message['Contents'] and '://' not in message['Contents'])]


def lowercase_member_data(member_data):
    member_data_copy = [message.copy() for message in member_data]
    for message in member_data_copy:
        message['Contents'] = message['Contents'].lower()
    
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


def count_messages_equaling_content(data, content):
    return len([message for message in data if content == message['Contents']])


if __name__ == "__main__":
    members = ['carman', 'dzss', 'friendlynoob', 'genzi', 'heibunny', 'jmoreojm', 'msn', 'rmj', 'tcray', 'willywonka']

    members_data = [load_json(f'data/uai_data/uai/{member}.json') for member in members]

    # Filter data
    members_data = [remove_short_messages(data) for data in members_data]
    members_data = [remove_long_messages(data) for data in members_data]
    members_data = [remove_contents_with_link(data) for data in members_data]
    members_data = remove_messages_in_common(members_data)

    for member_data in members_data:
        print(len(member_data))
