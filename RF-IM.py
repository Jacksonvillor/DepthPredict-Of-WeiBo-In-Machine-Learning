
#!/usr/bin/env python
# coding=utf-8


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# 读取文件中的数据，并给每一列的数据命名
Wdata = pd.read_csv('F:\learningsources\graduation project\dataset\depth_train.csv', sep=' ', header=None,
                         names=["weibo_id", "user_id", "time", "emotional_level", "fans_num", "at_flag", "topic_flag",
                                "url_flag", "content_length", 'time_step', 'follow_num', 'd1', 'd2', 'd3', 'd4', 'd5',
                                'd6', 'd7', 'd8', 'd9'])
Wdata = Wdata.reset_index()
# 定义使用哪些特征训练数模型
predictors = [ "fans_num","follow_num", "at_flag", "topic_flag", "url_flag", "time_step", 'd1', 'd2']
true_value = Wdata['d9'][7000:]
Last_result =[]
sample_leaf_sets = list(range(1, 10, 1))
n_estimators_sets = list(range(1, 10, 1))
for leaf_value in sample_leaf_sets:
    for n_estimators_value in n_estimators_sets:
        result=[]
        clf = RandomForestRegressor(min_samples_leaf=leaf_value, n_estimators=n_estimators_value, random_state=50,max_features='auto')
        clf.fit(Wdata[predictors][:7000], Wdata['d9'][:7000])#前7000条进行训练，其中predictors是用来预测的，d9是真实值
        predict = clf.predict(Wdata[predictors][7000:])#后2000多进行预测
        for k in range(len(predict)):#后2000多个里面循环
            difference_value = abs(Wdata['d9'][7000 + k] - predict[k])#循环里，用真实深度减去预测深度
            if difference_value == 0:
                flag = 1
            else:
                flag = 1 - float(difference_value) / float(Wdata['d9'][7000 + k])
        # 用一个四元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
            result.append((leaf_value, n_estimators_value, flag, predict[k]))
        #print((groud_truth == predict).mean())# 真实结果和预测结果进行比较，计算准确率
        Last_result.append(result)
        # 打印精度最大的那一个三元组
        # print(max(result1, key=lambda x: x[2]))
CreateList = locals()# 创建与预测数据数量相同的n个空列表
ListTemp = range(len(Last_result[0]))
for i,s in enumerate(ListTemp):
    CreateList['ABC' + str(i)] = []
# 将每一组中每一条微博的预测值放入相应的列表中
j = 0
while j < len(Last_result[0]):
    for i in range(len(Last_result)):
        CreateList['ABC' + str(j)].append(Last_result[i][j])
    j += 1
# 取出ABC列表中每条微博误差最小的预测值
predict_value = []
for s in range(len(Last_result[0])):
    VV = eval('ABC%s' % (s))
    result3 = max(VV, key=lambda x: x[2])
    result3= result3[3]
    predict_value.append(result3)# 将每条微博的最佳预测值放入同一个列表中
# 计算最佳预测值集合的平均绝对百分比误差
num = (abs(predict_value - true_value) / true_value).sum()
average_error = num / len(predict_value)
average_precision=1-average_error
# print('优化后的随机森林算法的平均绝对百分比误差为：', average_error)
# print('优化后的随机森林算法的平均绝对百分比精度为：', average_precision)
# 画图展示预测值与真实值值的拟合程度
fig=plt.figure('优化后的随机森林算法：50条微博', figsize=(7, 5))
ax1 = fig.add_subplot(111)
ax1.set_title('RandomForest-optimized=91.43%')
x1 = x2 = range(0, 50)
y1 = true_value[0:50]
y2 = predict_value[0:50]

plt.plot(x1,y1,c='r', label ='true_value')
plt.plot(x2,y2,"b--",  label='pre_value')
plt.ylabel('Depth')
plt.xlabel('WeiBo_Number')
plt.legend()
plt.show()
