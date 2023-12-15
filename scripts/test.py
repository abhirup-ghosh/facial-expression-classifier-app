import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://i.pinimg.com/originals/6e/cd/8a/6ecd8a85b2459b07d2889667aa9b6a0c.jpg'}

result = requests.post(url, json=data).json()
print(result)