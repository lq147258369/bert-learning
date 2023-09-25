import matplotlib.pyplot as plt # 画图工具
from sklearn import datasets # 机器学习库
from sklearn.preprocessing import LabelEncoder
from GeocodingCHN import Geocoding
from sklearn import tree
import numpy as np # 快速操作结构数组的工具
import pandas as pd # 数据分析处理工具
df=pd.read_csv('./data_sample_res.csv')

#合包完成时间-入库时间
df['pick_diff']=(pd.to_datetime(df['real_pick_finish_time'])-pd.to_datetime(df['enter_time']))/ np.timedelta64(1, 'h')
#合包完成时间-上架时间
df['hebao_finish_diff']=(pd.to_datetime(df['real_merge_finish_time'])-pd.to_datetime(df['real_on_shelf_time']))/ np.timedelta64(1, 'h')
#合包完成时间小时
df['hebao_finish_hour']=pd.to_datetime(df['real_merge_finish_time']).dt.hour
#标签：拣货时间-合包完成时间
df['label'] = (pd.to_datetime(df['real_pick_finish_time'])-pd.to_datetime(df['real_merge_finish_time']))/ np.timedelta64(1, 'h')

df_train=df[['pick_diff','hebao_finish_diff','hebao_finish_hour']]
df_label = df[['label']]

df_train=df_train.fillna(0)
df_label=df_label.fillna(0)

# 构建决策树
clf = tree.DecisionTreeRegressor()
clf.fit(df_train, df_label)
print(clf)



# 使用决策树进行预测
result = clf.predict([[89.930000,70.533611,15]])    # 输入也必须是数字的。分别代表了每个数字所代表的属性的字符串值
print(result)

# 将决策树保存成图片
from six import StringIO
import pydotplus

attr_names = ['pick_diff', 'hebao_finish_diff', 'hebao_finish_hour']   #特征属性的名称

dot_data = StringIO()
target_name=['None','Basic','Premium']
tree.export_graphviz(clf, out_file=dot_data,feature_names=attr_names,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
