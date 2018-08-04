#!/usr/bin/env python
# coding=utf-8

from collections import defaultdict
import pandas as pd
import jieba


#定义否定词列表
notdict = ['不','没','无','非','莫','弗', '勿', '毋', '未', '否', '别', '無', '休', '难道']
#读取微博源数据集
weibo_data = pd.read_csv('F:/learningsources/graduation project/dataset/weibo.train',sep='_001_',header=None,names=["weibo_id","user_id","time","emotional_level, engine=null"])
#定义停用词列表
stop_list = [' ']
#读取停用词词典
with open('F:/learningsources/graduation project/dataset/stopwords.txt','rb') as f:
    for line in f.readlines():
        line = line.decode('utf-8').strip('\r\n')
        stop_list.append(line)
#去掉停用词中的“@”和“#”号
stop_list.remove("@")
stop_list.remove("#")

#遍历转发文件寻找微博对应的转发次数    
def repostData():
    FF = {}
    with open('F:/learningsources/graduation project/dataset/Filed-1','rb') as Filed:
        Repost_data = Filed.readlines()
        for repost_data in Repost_data:
            repost_data1 = repost_data.decode('utf-8').strip('\r\n')
            repost_data2 = repost_data1.split(' ')
            FF[repost_data2[1]] = repost_data2[0]
            
    return FF


#计算情感值
def classifyWords(wordDict):
    #读取情感词典
    # (1) 情感词
    with open('F:/learningsources/graduation project/dataset/feel.txt','rb') as senlist:
        senList = senlist.readlines()
        senDict = defaultdict()
        for s in senList:
            s = s.decode('utf-8').strip('\r\n')
            if not s:
                continue
            Sen = s.split(' ')
            if len(Sen) < 2:
                continue
            senDict[Sen[0]] = Sen[1]
    # (2) 否定词
    notList = notdict
    
    #读取程度副词词典
    # (3) 程度副词
    with open('F:/learningsources/graduation project/dataset/level.txt','rb') as degreelist:
        degreeList = degreelist.readlines()
        degreeDict = defaultdict()
        for d in degreeList:
            d = d.decode('utf-8').strip('\r\n')
            if not d:
                continue
            Deg = d.split(',')
            if len(Deg) < 2:
                continue
            degreeDict[Deg[0]] = Deg[1]

   
    SenWord = []
    NotWord = []
    DegreeWord = []

#判断微博文本词语的属性
    for words in wordDict:
        senWord = defaultdict()
        notWord = defaultdict()
        degreeWord = defaultdict()
        for word in words:
            if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
                senWord[word] = senDict[word]
            elif word in notList and word not in degreeDict.keys():
                notWord[word] = -1
            elif word in degreeDict.keys():
                degreeWord[word] = degreeDict[word]
#将微博文本的词语归类，分别放入三个列表中
        SenWord.append(senWord)
        NotWord.append(notWord)
        DegreeWord.append(degreeWord)
        
    W = 1
    F = 0
    S = 0
    sumScore = 0
    senLoc = []
    notLoc = []
    degreeLoc = []
    
    for senWord in SenWord:
        senLoc.append(senWord.keys())
    for notWord in NotWord:
        notLoc.append(notWord.keys())
    for degreeWord in DegreeWord:
        degreeLoc.append(degreeWord.keys())
   


    for i in range(len(wordDict)):
        #第i条微博文本
        segresult = wordDict[i]
        for feel in segresult:
        # 如果该词为情感词
            if feel in senLoc[i]:
                S += float(SenWord[i][feel])
        # 如果为否定词
            elif feel in notLoc[i]:
                W *= -1
        # 如果为程度副词
            elif feel in degreeLoc[i]:
                F += float(DegreeWord[i][feel]) 
            else:
                pass
    if S == 0:
        S = 1
    if W == 0:
        W = 1
    if F == 0:
        F =1
    #计算微博文本总分数    
    sumScore = S * W * F
    #如果微博文本有多个句子，求其均值为最终情感分数
    Score = float(sumScore)/float(len(wordDict))
    return Score

 #定义微博内容中是否包含“@”函数   
def atflagHandler(wordDict):
    at_flag = 0
    for i in  range(len(wordDict)):
        segresult = wordDict[i]
        for word in segresult:
        # 如果有'@'
            if word in ['@']:
                at_flag += 1
    return at_flag
 #定义微博内容中是否包含“#”函数  
def topicflagHandler(wordDict):
    topic_flag = 0
    for i in  range(len(wordDict)):
        segresult = wordDict[i]
        for word in segresult:
        # 如果有'#'
            if word in ['#']:
                topic_flag += 1
    return topic_flag

#定义微博内容中是否包含url函数     
def urlhandler(wordDict):
    url_num = 0
    for i in  range(len(wordDict)):
        segresult = wordDict[i]
        for word in segresult:
        # 如果有'url'
            if word in ['http']:
                url_num += 1
    return url_num
 #定义计算微博内容长度函数   
def lengthhandler(wordDict):
    content_length = 0
    for i in  range(len(wordDict)):
        cl = len(wordDict[i])
        content_length += cl
    return content_length

#定义特征相关列的名称
weibo_data['follow_num'] = 0
weibo_data['fans_num'] = 0          
weibo_data['at_flag'] = 0
weibo_data['topic_flag'] = 0
weibo_data['url_flag'] = 0
weibo_data['content_length'] = 0
weibo_data['time_step'] = 0
weibo_data['depth1'] = 0
weibo_data['depth2'] = 0
weibo_data['depth3'] = 0
weibo_data['depth4'] = 0
weibo_data['depth5'] = 0
weibo_data['depth6'] = 0
weibo_data['depth7'] = 0
weibo_data['depth8'] = 0
weibo_data['depth9'] = 0        
#遍历处理每一条微博
for i in range(len(weibo_data)):
    train_data = []
    Newsent = []
    #将微博内容以句号分割开处理
    Sentence = weibo_data.values[i][3].split('。')
    #定义微博ID与用户ID的取值
    weiboId = weibo_data.values[i][0]
    userId = weibo_data.values[i][1]
    #将分割开的句子用jieba分词工具分词
    for sentence in Sentence:
        segresult = list(jieba.cut(sentence))
        newsent = []
        for word in segresult:
            if word in stop_list:
                    continue
            else:
                newsent.append(word)
        #将一条微博多个句子的结果全部存入newsent列表中
        Newsent.append(newsent)
    #调用计算情感值的函数
    emotional_level = classifyWords(Newsent)
    #修改该微博对应的列的情感值，i为第i条微博，3为第3列
    weibo_data.iat[i,3] = emotional_level 
     
   #遍历用户关系文件计算用户ID对应的关注数
    for line in open('F:/learningsources/graduation project/dataset/users.txt','rb'):
        line = line.decode('utf-8').strip('\r\n')
        Line = line.split(" ")
        if userId != int(Line[0]):
            continue
        else:
            follow_num = int(Line[1])
            weibo_data.iat[i,4] = follow_num
     #遍历粉丝关系文件计算用户ID对应的关注数                      
    for line in open('F:/learningsources/graduation project/dataset/fansNum.txt','rb'):
        line = line.decode('utf-8').strip('\r\n')
        Line = line.split(',')
        if userId != int(Line[0]):
            continue
        else:
            fans_num = int(Line[1])
            weibo_data.iat[i,5] = fans_num
     #调用计算微博内容包含“@“的函数           
    at_Flag = atflagHandler(Newsent)
    weibo_data.iat[i,6] = at_Flag
    
    #调用计算微博内容包含“#“的函数
    topic_Flag = topicflagHandler(Newsent)
    weibo_data.iat[i,7] = topic_Flag
     
                  
     #调用计算微博内容包含url的函数
    url_flag = urlhandler(Newsent)
    weibo_data.iat[i,8] = url_flag

    #调用计算微博内容长度的函数
    content_length = lengthhandler(Newsent)
    weibo_data.iat[i,9] = content_length

    #计算每一条微博发布的时间段
    tt = weibo_data['time'][i].split(':')
    if int(tt[0]) in range(7):
        time_step = 0
    if int(tt[0]) in range(7,13):
        time_step = 1
    if int(tt[0]) in range(13,19):
        time_step = 2
    if int(tt[0]) in range(19,24):
        time_step = 3
    
    weibo_data.iat[i,10] = time_step


    #计算每个时间点微博的传播深度            
    for line in open('F:/learningsources/graduation project/dataset/repostDepthCount.txt','rb'):
        line = line.decode('utf-8').strip('\r\n')
        Line = line.split(',')
        if weiboId == int(Line[0]):
            weibo_data.iat[i,11] = int(Line[2])
            weibo_data.iat[i,12] = int(Line[4])
            weibo_data.iat[i,13] = int(Line[12])
            weibo_data.iat[i,14] = int(Line[20])
            weibo_data.iat[i,15] = int(Line[28])
            weibo_data.iat[i,16] = int(Line[40])
            weibo_data.iat[i,17] = int(Line[96])
            weibo_data.iat[i,18] = int(Line[192])
            weibo_data.iat[i,19] = int(Line[288])
                          
    
    
            
print(weibo_data)
#将结果保存到文件中
weibo_data.to_csv('F:/learningsources/graduation project/dataset/5.csv',index=False,header=False,sep=' ')

