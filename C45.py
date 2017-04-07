import pandas as pd
import math


class ThresholdDict(dict):
    def __init__(self, data=None, threshold=None, **kwargs):
        self._data = {}
        self._threshold = threshold
        if data is None:
            data = {}
        self._data.update(**kwargs)
        self.great = None
        self.less = None

    def __setitem__(self, key, value):
        if not self._threshold:
            self._data[key] = value
            return
        if key > self._threshold:
            self.great = value
        else:
            self.less = value

    def __getitem__(self, key):
        if not self._threshold:
            return self._data[key]
        if key > self._threshold:
            return self.great
        return self.less

    def __bool__(self):
        return bool(self._data) or bool(self._threshold)

    def items(self):
        items = list(self._data.items())
        if self._threshold:
            items.extend([('>{}'.format(self._threshold), self.great), ('<={}'.format(self._threshold), self.less)])
        return items

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    def is_continuous(self):
        return bool(self._threshold)


class C45(object):
    class Node(object):
        def __init__(self, _id):
            self.id = _id
            self.attribute = None
            self.branches = ThresholdDict()

        def __str__(self):
            return self.attribute

        __repr__ = __str__

        @property
        def node_name(self):
            # 可视化时，每个node，必须要有独一无二的name
            return ''.join([self.attribute, str(self.id)])

        def add_branch_node(self, value, node):
            self.branches[value] = node

        def add_continuous_node(self, threshold, great, less):
            self.branches.threshold = threshold
            self.branches.great = great
            self.branches.less = less

        def add_to_graph(self, graph):
            graph.node(self.node_name, self.__str__())
            for edge_name, branch_node in self.branches.items():
                branch_node.add_to_graph(graph)
                graph.edge(self.node_name, branch_node.node_name, label=str(edge_name))

    def __init__(self, data, target, continuous=None):
        """ 初始化 ID3

        :param data: DataFrame，训练数据集
        :param target: 训练数据集中的目标attribute
        """
        self.node_counter = 0 # 方便可视化
        self.root_node = None
        self.data = data
        self.target = target
        self.attribute_values = {}
        self.continuous = continuous
        if not self.continuous:
            self.continuous = []
        for attribute in data.columns:
            self.attribute_values[attribute] = data[attribute].unique()
        self.max_depth = math.inf

    def _entropy(self, data, attribute):
        """计算某个attribute的熵，私有
        """
        value_freq = data[attribute].value_counts()
        data_entropy = 0.0
        N = len(data)
        #  $\mathrm{Entropy(A)}=\sum_{i=1}^{s} \mathrm{Info(data[A])}$
        for value, freq in value_freq.items():
            p = freq / N
            data_entropy += p * self._info(data, attribute, value)
        return data_entropy

    def _info(self, data, attribute=None, attribute_value=None):
        if attribute and attribute_value:
            data = data[data[attribute] == attribute_value]
        target_value_freq = data[self.target].value_counts()
        data_info = 0.0
        N = len(data)
        # $\mathrm{Info}=\sum_{i=1}^{c} - p_i \log_2 p_i$
        for freq in target_value_freq.values:
            p = freq / N
            data_info -= p * math.log(p, 2)
        return data_info

    @staticmethod
    def _split_info(data, attribute):
        freq = data[attribute].value_counts()
        split_info = 0.0
        N = len(data)
        for freq in freq.values:
            p = freq / N
            split_info -= p * math.log(p, 2)
        return split_info

    def _threshold_info(self, data, attribute, threshold):
        temp_data = data.copy()
        temp_data['split'] = temp_data[attribute] > threshold
        return self._entropy(temp_data, 'split')

    def _find_threshold(self, data, attribute):
        temp_data = data.copy()
        temp_data = temp_data.sort_values(attribute).reset_index(drop=True)
        temp_data['diff'] = temp_data[attribute].diff()

        last_row = None
        min_entropy = math.inf
        chosen_threshold = None
        for index, row in temp_data.iterrows():
            if index == 0:
                last_row = row
            else:
                if row[attribute] != last_row[attribute]:
                    threshold = last_row[attribute]
                    temp_entropy = self._threshold_info(temp_data, attribute, threshold)
                    if temp_entropy < min_entropy:
                        min_entropy = temp_entropy
                        chosen_threshold = threshold
                last_row = row
        return min_entropy, chosen_threshold

    def _make_decision_tree(self, data, node, depth):
        """建立决策树，私有
        """
        # 当数据都为同一类数据时，直接返回
        if len(data[self.target].value_counts()) == 1:
            node.attribute = data[self.target].value_counts().index[0]
            return
        # 如果除了target外，已经没有其他attribute了，那也返回
        if (len(data.columns) == 1) or (depth == self.max_depth):
            print(data[self.target].value_counts())
            node.attribute = data[self.target].value_counts().argmax()
            return

        not_valid_attributes = []
        for attribute in data.columns:
            if len(data[attribute].value_counts()) == 1:
                not_valid_attributes.append(attribute)
        data = data.drop(not_valid_attributes, axis=1)

        data_entropy = self._info(data)
        max_gain_ratio = -math.inf
        threshold = None
        for attribute in data.columns:
            if attribute == self.target:
                continue
            if attribute in self.continuous:
                entropy, temp_threshold = self._find_threshold(data, attribute)
                if not temp_threshold:
                    continue
                temp_gain_ratio = (data_entropy - entropy)
            else:
                temp_gain_ratio = (data_entropy - self._entropy(data, attribute))
            if math.isclose(self._split_info(data, attribute), 0):
                temp_gain_ratio = math.inf
            else:
                temp_gain_ratio = temp_gain_ratio / self._split_info(data, attribute)

            if temp_gain_ratio > max_gain_ratio:
                max_gain_ratio = temp_gain_ratio
                node.attribute = attribute
                if attribute in self.continuous:
                    threshold = temp_threshold

        if node.attribute in self.continuous:
            great_data = data[data[node.attribute] > threshold]
            less_data = data[data[node.attribute] <= threshold]
            great_node = self._new_node()
            less_node = self._new_node()
            node.add_continuous_node(threshold, great_node, less_node)
            self._make_decision_tree(great_data, great_node, depth+1)
            self._make_decision_tree(less_data, less_node, depth+1)
        else:
            for value in data[node.attribute].value_counts().index:
                branch_data = data[data[node.attribute] == value]
                branch_data = branch_data.drop(node.attribute, axis=1)
                branch_node = self._new_node()
                node.add_branch_node(value, branch_node)
                self._make_decision_tree(branch_data, branch_node, depth+1)

    def run(self, max_depth=math.inf):
        self.max_depth = max_depth
        self.root_node = self._new_node()
        self._make_decision_tree(self.data, self.root_node, 0)

    def _new_node(self):
        self.node_counter += 1
        return self.Node(_id=self.node_counter)

    def render_decision_tree(self, filename):
        """渲染决策树，需要graphviz支持
        """
        if not self.root_node:
            raise ValueError('Tree not decided!')

        from graphviz import Digraph
        dot_graph = Digraph(comment="Decision Tree")
        self.root_node.add_to_graph(dot_graph)
        dot_graph.render(filename)

    def _predict(self, row, node, force):
        """实际进行递归预测的函数，私有
        """
        if node.branches:
            try:
                if node.attribute in self.continuous:
                    if row[node.attribute] > node.branches.threshold:
                        return self._predict(row, node.branches.great, force)
                    return self._predict(row, node.branches.less, force)
                else:
                    return self._predict(row, node.branches[row[node.attribute]], force)
            except KeyError as e:
                if force:
                    return self._predict(row, next(iter(node.branches.values())), force)
                # TODO
                raise e
        else:
            return node.attribute

    def predict(self, data, force=False):
        """预测测试数据集的target值

        :param data: DataFrame 测试数据集
        :param force: 默认为 False，如果为 True，则当branch都不符合测试数据的值时，任意挑选一个branch
        :return: 预测值，list
        """
        if not hasattr(data, 'iterrows'):
            data = pd.DataFrame([data])
        results = []
        for index, row in data.iterrows():
            results.append(self._predict(row, self.root_node, force=force))
        return results

    @staticmethod
    def score(predict_results,  actual_results):
        return sum(predict_results == actual_results) / len(predict_results)

if __name__ == '__main__':
    train_data = pd.read_csv('golf_c.csv')
    C45_solver = C45(train_data, target='play', continuous=['humidity'])
    C45_solver.run()
    C45_solver.render_decision_tree('./dtree')
    test_data = train_data
    result = test_data['play'].values
    test_data.drop('play', axis=1, inplace=True)
    predict = C45_solver.predict(test_data)
    accuracy = C45_solver.score(predict, result)
    print('The accuracy of the prediction of test data is {}'.format(accuracy))

