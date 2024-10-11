import pickle

with open('29.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)