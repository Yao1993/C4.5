from C45 import C45
import pandas as pd
import numpy as np


np.random.seed(1993)
data = pd.read_csv('./balance-scale/balance-scale.csv')
permutation = np.random.permutation(len(data))[:30]
test_data = data.iloc[permutation].copy()
train_data = data.drop(permutation)
C45_solver = C45(train_data, target='Class', continuous=['Left-Weight', 'Left-Distance',
                                                         'Right-Weight', 'Right-Distance'])
C45_solver.run(40)
C45_solver.render_decision_tree('./balance-scale/dtree')
result = test_data['Class'].values
test_data.drop('Class', axis=1, inplace=True)
predict = C45_solver.predict(test_data)
accuracy = C45_solver.score(predict, result)
print('The accuracy of the prediction of test data is {}'.format(accuracy))
