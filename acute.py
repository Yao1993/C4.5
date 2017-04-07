from C45 import C45
import pandas as pd
import numpy as np


def train(target, drop, tree_name):
    data = pd.read_csv('./acute/diagnosis.csv')
    data.drop(drop, axis=1, inplace=True)
    permutation = np.random.permutation(len(data))[:10]
    test_data = data.iloc[permutation].copy()
    train_data = data.drop(permutation)
    C45_solver = C45(train_data, target=target, continuous=['Temperature'])
    C45_solver.run()
    C45_solver.render_decision_tree('./acute/'+tree_name)
    result = test_data[target].values
    test_data.drop(target, axis=1, inplace=True)
    predict = C45_solver.predict(test_data)
    accuracy = C45_solver.score(predict, result)
    print('The accuracy of the prediction of test data is {}'.format(accuracy))


train('decision1', 'decision2', 'dtree1')
train('decision2', 'decision1', 'dtree2')