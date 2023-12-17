import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/0/09/The_joy_of_the_happy_face_by_Rasheedhrasheed.jpg'}

result = requests.post(url, json=data).json()
print(result)