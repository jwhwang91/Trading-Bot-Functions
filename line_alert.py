import requests

def Sendmessage(msg):
    try:
        TARGET_URL = 'https://notify-api.line.me/api/notify'
        TOKEN = 'KOWIhJlkcJMEKH6Pv97emg42CTEzdEIJwRItdw5OmXx'
        # 요청합니다.
        response = requests.post(
            TARGET_URL,
            headers={
            'Authorization': 'Bearer ' + TOKEN
            },
            data={
                'message': msg
                }
            )
    except Exception as ex:
        print(ex)
