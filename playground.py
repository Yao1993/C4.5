import pandas as pd


def test(data):
    temp_data = data
    temp_data['hahah'] = data['play']
    print(id(data))
    print(id(temp_data))

train_data = pd.read_csv('golf_c.csv')
test(train_data)
print(train_data)