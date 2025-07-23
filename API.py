import requests

def get_data():

    try:
        response = requests.get("https://datasets-server.huggingface.co/rows?dataset=mmathys%2Fopenai-moderation-api-evaluation&config=default&split=train&offset=0&length=100")

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print('Error: ', response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print('Error: ', e)
        return None

def get_seconde_data():
    try:
        response = requests.get("https://datasets-server.huggingface.co/rows?dataset=JailbreakBench%2FJBB-Behaviors&config=behaviors&split=harmful&offset=0&length=100")

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print('Error: ', response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print('Error: ', e)
        return None