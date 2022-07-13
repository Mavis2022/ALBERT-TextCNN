###  预处理预测数据
import pandas as pd
predict_data = pd.read_csv('data/dy_cate_golden_set_v2.csv', dtype = {'spu_id':'string'})
pd.DataFrame(predict_data['spu_name']).to_csv('data/golden_set_to_pred_v2.txt',header=None,index=False)
