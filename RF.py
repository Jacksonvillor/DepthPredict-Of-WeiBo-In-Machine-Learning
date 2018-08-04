#!/usr/bin/env python
# coding=utf-8

# 决策树算法
import pandas as pd
from boto import sns
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
# 读取文件数据
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

Wdata = pd.read_csv('F:\learningsources\graduation project\dataset\depth_train.csv', sep=' ', header=None,
                    names=["weibo_id", "user_id", "time", "emotional_level", "fans_num", "at_flag", "topic_flag",
                           "url_flag", "content_length", 'time_step', 'follow_num', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
                           'd7', 'd8', 'd9'])
Wdata = Wdata.reset_index()
# 定义训练模型式时使用的特征
predictors = [ "fans_num", "at_flag", "topic_flag", "url_flag",  "time_step", "d1",
              "d2"]

# 定义训练数据的自变量个目标变量
train_x = Wdata[predictors][:7000]
train_y = Wdata['d9'][:7000]

# 定义测试数据的自变量和目标变量
groud_truth = Wdata[predictors][7000:]
true_value = Wdata['d9'][7000:]

rfc = RandomForestClassifier(max_features='auto', max_depth=4, random_state=50)
tuned_parameters = [{'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9]}]
# cv设置交叉验证
clf = GridSearchCV(estimator=rfc, param_grid=tuned_parameters, cv=5, n_jobs=1)
# 拟合训练集
clf.fit(train_x, train_y)
RFC_best=clf.best_params_
print('最佳参数是:')
print(clf.best_params_)
# 打印最佳得分
print('模型的最佳得分是：', clf.best_score_)


# 建立模型
a1 = clf.best_params_["n_estimators"]
a2 = clf.best_params_["min_samples_leaf"]
clf = RandomForestClassifier(n_estimators=a1, min_samples_leaf=a2)
# 训练模型
clf = clf.fit(train_x, train_y)
# 模型预测
pre_value = clf.predict(groud_truth)

# 计算平均绝对百分比误差
a = (abs(pre_value - true_value) / true_value).sum()
average_error = a / len(pre_value)
average_precision=1-average_error
# print('随机森林算法的平均绝对百分比误差为：', average_error)
# print('随机森林算法的平均绝对百分比精度为：', average_precision)
#

# 画图展示预测值与真实值的拟合程度
fig=plt.figure('随机森林算法：50条微博', figsize=(7, 5))
ax1 = fig.add_subplot(111)
ax1.set_title('RandomForest_average_precision=70.73%')
x1 = x2 = range(0, 50)
y1 = true_value[0:50]
y2 = pre_value[0:50]
plt.plot(x1,y1,c='r', label ='true_value')
plt.plot(x2,y2,"b--",  label='pre_value')
plt.ylabel('Depth')
plt.xlabel('WeiBo_Number')
plt.legend()
# plt.savefig('F:/learningsources/graduation project/result_images/RF/RF_LINE.png')
plt.show()

