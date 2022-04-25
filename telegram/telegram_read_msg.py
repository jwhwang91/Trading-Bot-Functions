
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

api_id = 15995215
api_hash = '2acbb36d5ec1413f456cbec3b7590147'
phone = "+821057751863"
username = 'jwhwang91'
chat = 'https://t.me/+ROgBmJXQsB1jMTQ9'

client = TelegramClient(username, api_id, api_hash)
client.start()
print("Client Created")
# Ensure you're authorized
if not client.is_user_authorized():
    client.send_code_request(phone)
    try:
        client.sign_in(phone, input('Enter the code: '))
    except SessionPasswordNeededError:
        client.sign_in(password=input('Password: '))

user_input_channel = input('ROgBmJXQsB1jMTQ9')

if user_input_channel.isdigit():
    entity = PeerChannel(int(user_input_channel))
else:
    entity = user_input_channel

my_channel = client.get_entity(entity)

from telethon.tl.functions.messages import (GetHistoryRequest)
from telethon.tl.types import (
PeerChannel
)

offset_id = 0
limit = 100
all_messages = []
total_messages = 0
total_count_limit = 0

while True:
    print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
    history = client(GetHistoryRequest(
        peer=my_channel,
        offset_id=offset_id,
        offset_date=None,
        add_offset=0,
        limit=limit,
        max_id=0,
        min_id=0,
        hash=0
    ))
    if not history.messages:
        break
    messages = history.messages
    for message in messages:
        all_messages.append(message.to_dict())
    offset_id = messages[len(messages) - 1].id
    total_messages = len(all_messages)
    if total_count_limit != 0 and total_messages >= total_count_limit:
        break