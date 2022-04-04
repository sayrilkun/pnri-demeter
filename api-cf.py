import requests

url = "https://v1.nocodeapi.com/sayrilkun/fbsdk/bdwzesmLPRNWrDXq/firestore/allDocuments?collectionName=Hoya/"
params = {}
r = requests.get(url = url, params = params)
result = r.json()
# print(result)

# for i in range(len(result)):
#     name = result[i]["_fieldsProto"]
#     print(name)

name = result[0]["_fieldsProto"]
print(name)

