
class MyModel:
    def __init__(self):
        self.xgb_r_param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'eval_metric': 'rmse', 'max_leaf_nodes': 10}
        self.xgb_c_param = {'objective': 'multi:softmax', 'num_class': 2, 'max_depth': 6, 'eta': 0.05, 'silent': 1,
             'eval_metric': 'merror', 'max_leaf_nodes': 10}

        self.xgb_r_num_round = 65
        self.xgb_c_num_round = 80