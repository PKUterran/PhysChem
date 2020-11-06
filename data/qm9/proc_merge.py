import json
import os

csv2json = {}

for file in os.listdir('mid'):
    fp = open(f'mid/{file}')
    d: dict = json.load(fp)
    fp.close()
    csv2json.update(d)

print(len(csv2json.items()))
fp = open('csv2json.json', 'w+')
json.dump(csv2json, fp)
fp.close()
