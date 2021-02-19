import pickle
import json

with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

new_data = {}

for k in network.keys():
    new_data[k] = network[k].tolist()

with open("sample_weight.json", 'w') as f:
    json.dump(new_data, f, indent=4)
