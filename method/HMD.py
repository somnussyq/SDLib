# coding:UTF-8

from baseclass.SDetection import SDetection
from tool import config
from random import choice,shuffle
import numpy as np
from tool.qmath import sigmoid
from math import log
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier


class HMD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]', k=None, n=None ):
        super(HMD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(HMD, self).readConfiguration()
        options = config.LineConfig(self.config['HMD'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.neighborNum = int(options['-N'])
        self.epoch = int(options['-ep'])
        self.rate = float(options['-r'])
        self.neg = int(options['-neg'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI = float(regular['-u']), float(regular['-i'])


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

        for user in self.dao.trainingSet_u:
            self.MUD[user] = 0
            for item in self.dao.trainingSet_u[user]:
                # 每个用户评分项目的流行度之和
                self.MUD[user] += len(self.dao.trainingSet_i[item])
                self.P[item] = len(self.dao.trainingSet_i[item])
            self.MUD[user]/float(len(self.dao.trainingSet_u[user]))


            # 用户流行度list
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
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
        for item, userS in self.dao.trainingSet_i.items():
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
        # 找到与用户A评分项目a评分流行度相同的项目b，再找评过项目b的用户B
        p5 = 'UIPU'
        #找到与用户A对产品a评分相同的用户B
        p6 = 'US'
        mPaths = [p5,p6]

        print 'Generating random meta-path random walks...'
        self.walks = []

        for user in self.dao.user:
            for mp in mPaths:
                if mp == p1:
                    self.walkCount = 5
                if mp == p2:
                    self.walkCount = 5
                if mp == p3:
                    self.walkCount = 5
                if mp == p4:
                    self.walkCount = 5
                if mp == p5:
                    self.walkCount = 5
                if mp == p6:
                    self.walkCount = 5

                for t in range(self.walkCount):
                    path = [(user, 'U')]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'

                    for i in range(self.walkLength / (len(mp[1:]))):
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

                                path.append((nextNode, tp))
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
        self.W = np.random.rand(self.dao.trainingSize()[0] + self.dao.testSize()[0], self.walkDim) / 10  # user
        self.G = np.random.rand(self.dao.trainingSize()[1] + self.dao.testSize()[1], self.walkDim) / 10  # item

        iteration = 1
        userList = self.dao.user.keys()
        itemList = self.dao.item.keys()

        while iteration <= self.epoch:
            self.loss = 0
            self.rLoss = 0
            self.wLoss = 0

            # Rating MF
            self.dao.ratings = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
            for user in self.dao.ratings:
                for item in self.dao.ratings[user]:
                    rating = self.dao.ratings[user][item]
                    error = rating - self.predictRating(user, item)
                    u = self.dao.all_User[user]
                    i = self.dao.all_Item[item]
                    p = self.W[u]
                    q = self.G[i]
                    self.rLoss += error**2
                    self.loss += self.rLoss

                    # update latent vectors
                    self.W[u] += self.rate * (error * q - self.regU * p)
                    self.G[i] += self.rate * (error * p - self.regI * q)

            for walk in self.walks:
                # i:从1开始的序号; node:路径中的每个节点
                for i, node in enumerate(walk):
                    #滑动窗口中的邻居
                    neighbors = walk[max(0, i - self.winSize / 2): min(len(walk) - 1, i + self.winSize / 2 )]
                    # center:(user/item)id; ctp:nodeType
                    center, ctp = walk[i]
                    if ctp == 'I' or ctp == 'P': #Item
                        centerVec = self.G[self.dao.all_Item[center]]
                    else: #User
                        centerVec = self.W[self.dao.all_User[center]]

                    for entity, tp in neighbors:
                        currentVec = ''
                        if tp == 'U' or tp == 'M' or tp == 'Q' or tp == 'R' or tp == 'S' and center <> entity:
                            currentVec = self.W[self.dao.all_User[entity]]
                            self.W[self.dao.all_User[entity]] +=  self.rate * (
                                1 - sigmoid(currentVec.dot(centerVec))) * centerVec
                            if ctp == 'U' or ctp == 'M' or ctp == 'Q' or ctp == 'R' or ctp == 'S':
                                self.W[self.dao.all_User[center]] +=  self.rate * (
                                    1 - sigmoid(currentVec.dot(centerVec))) * currentVec
                            else:
                                self.G[self.dao.all_Item[center]] += self.rate * (
                                    1 - sigmoid(currentVec.dot(centerVec))) * currentVec

                            self.wLoss += - log(sigmoid(currentVec.dot(centerVec)))
                            self.loss += self.wLoss

                            for i in range(self.neg):
                                sample = choice(userList)
                                while sample == entity:
                                    sample = choice(userList)
                                sampleVec = self.W[self.dao.all_User[sample]]
                                self.W[self.dao.all_User[sample]] -=  self.rate * (
                                    1 - sigmoid(-sampleVec.dot(centerVec))) * centerVec
                                if ctp == 'U' or ctp == 'M' or ctp == 'Q' or ctp == 'R' or ctp == 'S':
                                    self.W[self.dao.all_User[center]] -=  self.rate * (
                                        1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
                                else:
                                    self.G[self.dao.all_Item[center]] -= self.rate * (
                                        1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec

                        elif tp == 'I' or tp =='P' and center <> entity:
                            currentVec = self.G[self.dao.all_Item[entity]]
                            self.G[self.dao.all_Item[entity]] += self.rate * (
                                1 - sigmoid(currentVec.dot(centerVec))) * centerVec
                            if ctp == 'U' or ctp == 'M' or ctp == 'Q' or ctp == 'R' or ctp == 'S':
                                self.W[self.dao.all_User[center]] +=  self.rate * (
                                    1 - sigmoid(currentVec.dot(centerVec))) * currentVec
                            else:
                                self.G[self.dao.all_Item[center]] += self.rate * (
                                    1 - sigmoid(currentVec.dot(centerVec))) * currentVec

                            self.wLoss += -  log(sigmoid(currentVec.dot(centerVec)))
                            self.loss += self.wLoss

                            for i in range(self.neg):
                                sample = choice(itemList)
                                while sample == entity:
                                    sample = choice(itemList)
                                sampleVec = self.G[self.dao.all_Item[sample]]
                                self.G[self.dao.all_Item[sample]] -= self.rate * (
                                    1 - sigmoid(-currentVec.dot(centerVec))) * centerVec
                                if ctp == 'U' or ctp == 'M' or ctp == 'Q' or ctp == 'R' or ctp == 'S':
                                    self.W[self.dao.all_User[center]] -= self.rate * (
                                        1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
                                else:
                                    self.G[self.dao.all_Item[center]] -= self.rate * (
                                        1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
            shuffle(self.walks)
            print 'iteration:', iteration, 'loss:', self.loss, 'rLoss', self.rLoss, 'wLoss', self.wLoss
            iteration += 1

        print 'User embedding generated.'

        # # 保存到本地文件
        userVectors = open('HMD-Amazon-userVec' + self.foldInfo + '.pkl', 'wb')
        itemVectors = open('HMD-Amazon-itemVec' + self.foldInfo + '.pkl', 'wb')
        pickle.dump(self.W, userVectors)
        pickle.dump(self.G, itemVectors)
        userVectors.close()
        itemVectors.close()
        return self.W


    def initModel(self):
        self.feature()
        self.ISU()
        self.mPath_rWalk()
        # 训练用户嵌入向量
        print 'Generating user embedding...'
        #self.W = np.random.rand(self.dao.trainingSize()[0] + self.dao.testSize()[0], self.walkDim) / 10  # user
        #self.G = np.random.rand(self.dao.trainingSize()[1] + self.dao.testSize()[1], self.walkDim) / 10  # item
        self.skipGram()


    def buildModel(self):
        # {userID：userOrder} ---> {userOrder:userID}
        # userOrder = dict((v, k) for k, v in self.dao.all_User.iteritems())

        # # 加载嵌入向量
        pkl_user = open('HMD-Amazon-userVec' + self.foldInfo + '.pkl', 'rb')
        userVec = pickle.load(pkl_user)

        # 训练数据
        for user in self.dao.trainingSet_u:
            self.training.append(userVec[self.dao.all_User[user]])
            self.trainingLabels.append(self.labels[user])

        # 测试数据
        for user in self.dao.testSet_u:
            self.test.append(userVec[self.dao.all_User[user]])
            self.testLabels.append(self.labels[user])


    def predict(self):
        clfRF = RandomForestClassifier(n_estimators=8)
        clfRF.fit(self.training, self.trainingLabels)
        pred_labels = clfRF.predict(self.test)
        return pred_labels

    ###   代码需要优化的内容
    #     1. 没有评分的用户
    #     2. 一个用户的item邻居很少？怎么处理？