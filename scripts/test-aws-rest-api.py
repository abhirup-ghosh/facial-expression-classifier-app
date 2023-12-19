import requests

url = 'https://xowsdry1bc.execute-api.eu-north-1.amazonaws.com/Test/predict'

data = {'url': 'https://assets.telegraphindia.com/abp/2018/6/24/1606282415_5fbdecaf96376_abhirup-ghosh.jpg'}
#'https://upload.wikimedia.org/wikipedia/commons/0/09/The_joy_of_the_happy_face_by_Rasheedhrasheed.jpg'}

result = requests.post(url, json=data).json()
print(result)
