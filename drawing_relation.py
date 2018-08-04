
import pandas as pd
import matplotlib.pyplot as plt

# 读取文件中的数据
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
Wdata = pd.read_csv('F:\learningsources\graduation project\dataset\depth_train.csv', sep=' ', header=None,
                         names=["weibo_id", "user_id", "time", "emotional_level", "fans_num", "at_flag", "topic_flag",
                                "url_flag", "content_length", 'time_step', 'follow_num', 'd1', 'd2', 'd3', 'd4', 'd5',
                                'd6', 'd7', 'd8', 'd9'])
predictors = ["emotional_level", "fans_num", "at_flag", "topic_flag", "url_flag", "content_length", "time_step"]
#
#


# # 粉丝数与传播深度的关系
#
fig=plt.figure('粉丝数与传播深度的关系', figsize=(7, 5))
ax1 = fig.add_subplot(111)
ax1.set_title('fans_num&Depth')
x = Wdata['fans_num'][:50]
y = Wdata['d9'][:50]
plt.scatter(x, y, color='blue', marker='o')
plt.ylabel('Depth')
plt.xlabel('Fans_Number')
plt.show()
#
# # 用户关注的数与微博传播深度的关系散点图
fig=plt.figure('用户关注数与传播深度的关系', figsize=(7, 5))
ax1 = fig.add_subplot(111)
ax1.set_title('followers_num&Depth')
x = Wdata['follow_num'][:50]
y = Wdata['d7'][:50]
plt.scatter(x, y,color='red',marker='o')
plt.ylabel('Depth')
plt.xlabel('Followers_Number')
plt.show()

png4 = plt.figure()
# #x轴为微博条数，分别为50，100，500，1000，2000
x1=x2=x3=x4=x5=x6=[50,100,500,1000,2000,2787]
# #y轴为每个预测模型在特定时刻的预测准确率
# y1=[0.13,0.35,0.32,0.39,0.30,0.34]
# y2=[0.57,0.57,0.28,0.37,0.39,0.43]
# y3=[0.60,0.64,0.57,0.60,0.63,0.51]
# y4=[0.69,0.65,0.57,0.57,0.58,0.55]
# y5=[0.65,0.73,0.68,0.56,0.66,0.71]
y6=[0.88,0.89,0.85,0.86,0.85,0.91]
# plt.plot(x1,y1,'rs-',label='KNN')
# plt.plot(x2,y2,'b*-',label='GBDT')
# plt.plot(x3,y3,'gv-',label='LR')
# plt.plot(x4,y4,'ko-',label='SVM')
# plt.plot(x5,y5,'ms-',label='RF')
plt.plot(x6,y6,'yD-',label='RF-IM')
plt.ylabel('accurate rate')
plt.xlabel('weibo_num')
plt.legend()
plt.show()


#六种模型预测准确率对比图
png5 = plt.figure('六种模型预测准确率对比图',figsize=(7,5))
plt.xlabel('Algorithm')
plt.ylabel('average presicion')
labels = ['KNN','GBDT','LR','SVM','RF','RF-IM']
precision_list = [0.34,0.43,0.51,0.55,0.71,0.91]
for x,y in zip(range(len(precision_list)),precision_list):
    plt.text(x,y, '%.2f' % y,ha='center',va='bottom')
plt.bar(range(len(precision_list)),precision_list,tick_label=labels,width=0.5)
plt.show()

#去掉用户个体特征的模型预测准确率对比图
png6 = plt.figure('去掉用户个体特征的模型预测准确率对比图',figsize=(7,5))
plt.xlabel('Algorithm')
labels = ['KNN','GBDT','LR','SVM','RF','RF-IM']
precision_list = [0.32,0.42,0.49,0.51,0.69,0.90]
for x,y in zip(range(len(precision_list)),precision_list):
    plt.text(x,y, '%.2f' % y,ha='center',va='bottom')
plt.bar(range(len(precision_list)),precision_list,tick_label=labels,width=0.5)
plt.show()

#去掉微博文本特征模型预测准确率对比图
png7 = plt.figure('去掉微博文本特征模型预测准确率对比图',figsize=(7,5))
plt.xlabel('Algorithm')
plt.ylabel('average presicion')
labels = ['KNN','GBDT','LR','SVM','RF','RF-IM']
precision_list = [0.30,0.39,0.45,0.48,0.66,0.90]
for x,y in zip(range(len(precision_list)),precision_list):
    plt.text(x,y, '%.2f' % y,ha='center',va='bottom')
plt.bar(range(len(precision_list)),precision_list,tick_label=labels,width=0.5)
plt.show()


#去掉微博内容特征四种模型预测准确率对比图
png8 = plt.figure(figsize=(7,5))
plt.xlabel('Algorithm')
plt.ylabel('average presicion')
labels = ['KNN','GBDT','LR','SVM','RF','RF-IM']
precision_list = [0.22,0.30,0.38,0.44,0.63,0.89]
for x,y in zip(range(len(precision_list)),precision_list):
    plt.text(x,y, '%.2f' % y,ha='center',va='bottom')
plt.bar(range(len(precision_list)),precision_list,tick_label=labels,width=0.5)
plt.show()

