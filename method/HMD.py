# coding:UTF-8

from baseclass.SDetection import SDetection
from tool import config
from random import choice,shuffle
import numpy as np
from tool.qmath import sigmoid
from math import log
import cPickle as pickle
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import gensim.models.word2vec as w2v


class HMD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]', k=None, n=None ):
        super(HMD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(HMD, self).readConfiguration()
        options = config.LineConfig(self.config['HMD'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.walkCount = int(options['-c'])
        self.winSize = int(options['-w'])
        self.neighborNum = int(options['-N'])
        self.epoch = int(options['-ep'])
        self.rate = float(options['-r'])
        self.neg = int(options['-neg'])
        self.ratingClass = int(options['-rc'])
        self.degreeClass = int(options['-dc'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI, self.regR, self.regW = float(regular['-u']), float(regular['-i']),\
                                                     float(regular['-R']), float(regular['-W'])


    def printAlgorConfig(self):
        super(HMD, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['methodName'] + ':'
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80


    # 计算项目及用户特征
    def feature(self):
        # 用户流行度均值
        self.MUD = {}
        # 排序后的字典，以用户为键值{user：order}
        self.UOM = {}
        # 排序后的字典，以顺序为键值{order: user}
        self.OUM = {}

        # 用户流行度极差
        self.RUD = {}
        self.UOR = {}
        self.OUR = {}

        # 用户流行度上四分位数
        self.QUD = {}
        self.UOQ = {}
        self.OUQ = {}

        # 项目流行度
        self.P = {}
        # {Item:Order}
        self.IOP = {}
        # {Order:Item}
        self.OIP = {}

        self.dao.u = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
        self.dao.i = dict(self.dao.trainingSet_i, **self.dao.testSet_i)

        for user in self.dao.u:
            self.MUD[user] = 0
            for item in self.dao.u[user]:
                # 每个用户评分项目的流行度之和
                self.MUD[user] += len(self.dao.i[item])
                self.P[item] = len(self.dao.i[item])
            self.MUD[user]/float(len(self.dao.u[user]))


            # 用户流行度list
            lengthList = [len(self.dao.i[item]) for item in self.dao.u[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

        self.sMUD = sorted(self.MUD.items(), key=lambda item: item[1])
        self.sRUD = sorted(self.RUD.items(), key=lambda item: item[1])
        self.sQUD = sorted(self.QUD.items(), key=lambda item: item[1])
        self.sP = sorted(self.P.items(), key=lambda item: item[1])

        mOrder = 1
        for i in self.sMUD:
            self.UOM[i[0]] = mOrder
            self.OUM[mOrder] = i[0]
            mOrder += 1
        # print self.MUD
        # print '-------------------'
        # print self.UOM
        # print '```````````````````'
        # print self.OUM
        rOrder = 1
        for i in self.sRUD:
            self.UOR[i[0]] = rOrder
            self.OUR[rOrder] = i[0]
            rOrder += 1

        qOrder = 1
        for i in self.sQUD:
            self.UOQ[i[0]] = qOrder
            self.OUQ[qOrder] = i[0]
            qOrder += 1

        pOrder = 1
        for i in self.sP:
            self.IOP[i[0]] = pOrder
            self.OIP[pOrder] = i[0]
            pOrder += 1

    # 构造一个ISU {item:{score:[user1,user2]}}字典
    def ISU(self):
        self.ISUDict = {}
        # 遍历字典中的键值对
        for item, userS in self.dao.i.items():
            scoreUDict = {}
            for user, score in userS.items():
                # 当某评分已经出现时
                if scoreUDict.has_key(score):
                    scoreUDict[score].append(user)
                # 当某评分第一次出现时：
                else:
                    userList = []
                    userList.append(user)
                    scoreUDict[score] = userList
            self.ISUDict[item] = scoreUDict
        return self.ISUDict



    def pickNeighbor(self, user, UODict, OUDict):
        # 测试边界节点
        #self.testV = []
        # for i, order in UODict.items():
        #     if order < self.neighborNum/2 or order>(len(UODict)- self.neighborNum/2):
        #         print i, order
        #         self.textV.append(i)
        # print self.textV

        # 随机测试节点
        #
        # for i in range(0, 10):
        #     item = choice(OUDict.values())
        #     while item in self.testV:
        #         item = choice(OUDict.values())
        #     self.testV.append(item)
        # print self.testV

        userOrder =UODict[user]
        neighborList = []

        leftIndex = userOrder
        rightIndex = userOrder

        if userOrder >= self.neighborNum/2 and userOrder <= (len(UODict)-1-self.neighborNum/2):
            while leftIndex > (userOrder - self.neighborNum/2):
                leftIndex -= 1
                neighborID = OUDict[leftIndex]
                if not neighborID:
                    pass
                neighborList.append(neighborID)
            while rightIndex < (userOrder + self.neighborNum/2):
                rightIndex += 1
                neighborID = OUDict[rightIndex]
                if not neighborID:
                    pass
                neighborList.append(neighborID)

        elif userOrder < self.neighborNum/2:
            while leftIndex > 0:
                leftIndex -= 1
                neighborID = OUDict[leftIndex]
                neighborList.append(neighborID)
                if not neighborID:
                    pass
            while rightIndex < self.neighborNum:
                print user,rightIndex
                rightIndex += 1
                neighborID = OUDict[rightIndex]
                if not neighborID:
                    pass
                neighborList.append[neighborID]
        else:
            while leftIndex > len(UODict) - self.neighborNum:
                leftIndex -= 1
                neighborID = OUDict[leftIndex]
                if not neighborID:
                    pass
                neighborList.append(neighborID)

            while rightIndex < len(UODict):
                rightIndex += 1
                neighborID = OUDict[rightIndex]
                if not neighborID:
                    pass
                neighborList.append(neighborID)

        # 检测邻居数是否出现异常
        # if len(neighborList) <> neighborNum:
        #     print neighborList, userOrder

        # 从邻居中随机选择一个
        neighbor = choice(neighborList)
        if not neighbor:
            pass
        #print neighbor
        return neighbor



    # 构造元路径并随机游走
    def mPath_rWalk(self):
        # U:user, M:MUD, Q:QUD, R:RUD, I:item, P:项目流行度, S:评分
        p1 = 'UIU'
        p2 = 'UMM'
        p3 = 'UQQ'
        p4 = 'URR'
        # # 找到与用户A评分项目a评分流行度相同的项目b，再找评过项目b的用户B
        p5 = 'UIPU'
        #找到与用户A对产品a评分相同的用户B
        p6 = 'US'
        mPaths = [p1]

        print 'Generating random meta-path random walks...'
        self.walks = []

        for user in self.dao.all_User:
            for mp in mPaths:
            #     if mp == p1:
            #         self.walkCount = 10
            #     if mp == p2:
            #         self.walkCount = 5
            #     if mp == p3:
            #         self.walkCount = 5
            #     if mp == p4:
            #         self.walkCount = 5
            #     if mp == p5:
            #         self.walkCount = 10
            #     if mp == p6:
            #         self.walkCount = 10

                for t in range(self.walkCount):
                    path = ['U'+user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'

                    for i in range(self.walkLength / len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'M':
                                    nextNode = self.pickNeighbor(lastNode, self.UOM, self.OUM)
                                if tp == 'Q':
                                    nextNode = self.pickNeighbor(lastNode, self.UOQ, self.OUQ)
                                if tp == 'R':
                                    nextNode = self.pickNeighbor(lastNode, self.UOR, self.OUR)
                                if tp == 'I':
                                    # 从用户评过分的项目中随机选择一个
                                    nextNode = choice(self.dao.trainingSet_u[lastNode].keys())
                                if tp == 'P':
                                    # 返回相似流行度的项目
                                    nextNode = self.pickNeighbor(lastNode, self.IOP, self.OIP)
                                if tp == 'U':
                                    # 找到评论过这个项目的用户
                                    nextNode = choice(self.dao.trainingSet_i[lastNode].keys())

                                if tp == 'S':
                                    # 随机选择用户评过分的项目
                                    item = choice(self.dao.trainingSet_u[lastNode].keys())
                                    # 记录这个项目的评分
                                    score = self.dao.trainingSet_u[lastNode][item]
                                    # 随机选择一个与该项目有相同评分的用户
                                    nextNode = choice(self.ISUDict[item][score])
                                    #### 如果没有相同评分的情况

                                path.append(tp+nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break
                    if path:
                        self.walks.append(path)
        shuffle(self.walks)
        print 'walks:', len(self.walks)

    def predictRating(self, user, item):
        u = self.dao.all_User[user]
        i = self.dao.all_Item[item]
        return self.W[u].dot(self.G[i])

    def skipGram(self):
        # 随机高斯分布
        # self.W = (np.random.randn(self.dao.trainingSize()[0] + self.dao.testSize()[0], self.walkDim)+1)/2 /100 # user
        # self.G = (np.random.randn(self.dao.trainingSize()[1] + self.dao.testSize()[1], self.walkDim)+1)/2 /100 # item
        self.W = np.random.rand(self.dao.trainingSize()[0] + self.dao.testSize()[0], self.walkDim) / 200
        self.G = np.random.rand(self.dao.trainingSize()[1] + self.dao.testSize()[1], self.walkDim) / 200

        # print 'training the rating matrix'
        # iteration = 1
        # while iteration <= self.epoch:
        #     self.loss = 0
        #
        #     # # Rating MF
        #     for user in self.dao.u:
        #         for item in self.dao.u[user]:
        #             rating = self.dao.u[user][item]
        #             error = rating - self.predictRating(user, item)
        #             u = self.dao.all_User[user]
        #             i = self.dao.all_Item[item]
        #             p = self.W[u]
        #             q = self.G[i]
        #             self.loss += self.regR * error ** 2
        #
        #             # update latent vectors
        #             self.W[u] += self.rate * (error * q - self.regU * p)
        #             self.G[i] += self.rate * (error * p - self.regI * q)

        print 'Generating user embedding...'
        model = w2v.Word2Vec(self.walks, sg=1, size=self.walkDim, window=4, min_count=0, iter=5)

        errorNumber = 0
        for user in self.dao.all_User:
            uid =self.dao.all_User[user]
            try:
                self.W[uid]= model.wv['U'+user]
            except (KeyError):
                self.W[uid]= np.random.randn(1,self.walkDim)
                errorNumber += 1



        print 'errorNumber',errorNumber
        print 'User embedding generated.'


        return self.W


    def initModel(self):
        self.feature()
        self.ISU()

        # 全局平均分字典 {item:meanRating} -->self.dao.itemMeans
        # 用户平均分字典 {user：meanRating} -->self.dao.userMeans
        # 平均分分为N级，建立分级字典{1级:{item1},... p级:{item}}
        # self.ratingDict = {}
        # ItemRatingDiff = max(self.dao.itemMeans.values()) - min(self.dao.itemMeans.values())
        # UserRatingDiff = max(self.dao.userMeans.valuse()) - min(self.dao.userMeans.values())


        # 全局流行度字典 {item：ratingCount}
        # 用户流行度字典 {user：ratingCount}
        # 流行度分为M级，建立分级字典{1级:{item},...q级:{item}}

        # 路径：user1-item1-item流行度级数-同级item2-给item2评过分的user2
        # 路径：user1-item1-itemMean-itemMean级数-同级item2-给item2评过分的user2
        # 路径：user1-userMean-userMean级数-同级userMean-user2
        # 路径：user1-ratingCont-ratingCount级数-同级userCount-user2



        self.mPath_rWalk()
        # # 训练用户嵌入向量



        # {userID：userOrder} ---> {userOrder:userID}
        self.userOrder = dict((v, k) for k, v in self.dao.all_User.iteritems())
        self.skipGram()


    def buildModel(self):
        # 训练数据
        for user in self.dao.trainingSet_u:
            self.training.append(self.W[self.dao.all_User[user]])
            self.trainingLabels.append(self.labels[user])

        # 测试数据
        for user in self.dao.testSet_u:
            self.test.append(self.W[self.dao.all_User[user]])
            self.testLabels.append(self.labels[user])


    def predict(self):
        clfRF = RandomForestClassifier(n_estimators=10)
        clfRF.fit(self.training, self.trainingLabels)
        pred_labels = clfRF.predict(self.test)
        # clfGBDT = GradientBoostingClassifier(n_estimators=10)
        # clfGBDT.fit(self.training, self.trainingLabels)
        # pred_labels = clfGBDT.predict(self.test)
        # clfDT = DecisionTreeClassifier(criterion='entropy')
        # clfDT.fit(self.training, self.trainingLabels)
        # pred_labels = clfDT.predict(self.test)
        return pred_labels

    ###   代码需要优化的内容
    #     1. 没有评分的用户
    #     2. 一个用户的item邻居很少？怎么处理？
    #     3. 异构的路径还需要仔细斟酌！
    #     4. 生成的向量直接放到分类器？是否还有其他方式？