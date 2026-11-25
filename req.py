import requests

URL = "https://hook.eu1.make.com/lzaevn47rmi4ip08r90411cvku1jqywj"


data = {
    "city": "Chennai"
}

res = requests.post(URL, json = data)

print(res.status_code)
