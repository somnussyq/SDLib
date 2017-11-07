# coding:UTF-8
from baseclass.SDetection import SDetection
from tool import config
from random import choice
import cPickle as pickle


class HMD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]', k=None, n=None ):
        super(HMD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(HMD, self).readConfiguration()
        options = config.LineConfig(self.config['HMD'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.neighborNum = int(options['-N'])


    def printAlgorConfig(self):
        super(HMD, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['methodName'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80


    # 计算项目及用户特征
    # def feature(self):
    #     # 用户流行度均值
    #     self.MUD = {}
    #     # 排序后的字典，以用户为键值{user：order}
    #     self.UOM = {}
    #     # 排序后的字典，以顺序为键值{order: user}
    #     self.OUM = {}
    #
    #     # 用户流行度极差
    #     self.RUD = {}
    #     self.UOR = {}
    #     self.OUR = {}
    #
    #     # 用户流行度上四分位数
    #     self.QUD = {}
    #     self.UOQ = {}
    #     self.OUQ = {}
    #
    #     # 项目流行度
    #     self.P = {}
    #     # {Item:Order}
    #     self.IOP = {}
    #     # {Order:Item}
    #     self.OIP = {}
    #
    #     for user in self.dao.trainingSet_u:
    #         self.MUD[user] = 0
    #         for item in self.dao.trainingSet_u[user]:
    #             # 每个用户评分项目的流行度之和
    #             self.MUD[user] += len(self.dao.trainingSet_i[item])
    #             self.P[item] = len(self.dao.trainingSet_i[item])
    #         self.MUD[user]/float(len(self.dao.trainingSet_u[user]))
    #
    #
    #         # 用户流行度list
    #         lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
    #         lengthList.sort(reverse=True)
    #         self.RUD[user] = lengthList[0] - lengthList[-1]
    #         self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]
    #
    #     self.sMUD = sorted(self.MUD.items(), key=lambda item: item[1])
    #     self.sRUD = sorted(self.RUD.items(), key=lambda item: item[1])
    #     self.sQUD = sorted(self.QUD.items(), key=lambda item: item[1])
    #     self.sP = sorted(self.P.items(), key=lambda item: item[1])
    #
    #     mOrder = 0
    #     for i in self.sMUD:
    #         self.UOM[i[0]] = mOrder
    #         self.OUM[mOrder] = i[0]
    #         mOrder += 1
    #     # print self.MUD
    #     # print '-------------------'
    #     # print self.UOM
    #     # print '```````````````````'
    #     # print self.OUM
    #     rOrder = 0
    #     for i in self.sRUD:
    #         self.UOR[i[0]] = rOrder
    #         self.OUR[rOrder] = i[0]
    #         rOrder += 1
    #
    #     qOrder = 0
    #     for i in self.sQUD:
    #         self.UOQ[i[0]] = qOrder
    #         self.OUQ[qOrder] = i[0]
    #         qOrder += 1
    #
    #     pOrder = 0
    #     for i in self.sP:
    #         self.IOP[i[0]] = pOrder
    #         self.OIP[pOrder] = i[0]
    #         pOrder += 1
    #
    #     # 保存字典到本地文件
    #     saveM = file('../midData/MUD.pkl', 'wb')
    #     pickle.dump(self.UOM, saveM, True)
    #     pickle.dump(self.OUM, saveM, True)
    #     saveM.close()
    #
    #     saveR = file('../midData/RUD.pkl', 'wb')
    #     pickle.dump(self.UOR, saveR, True)
    #     pickle.dump(self.OUR, saveR, True)
    #     saveR.close()
    #
    #     saveQ = file('../midData/QUD.pkl', 'wb')
    #     pickle.dump(self.UOQ, saveQ, True)
    #     pickle.dump(self.OUQ, saveQ, True)
    #     saveQ.close()
    #
    #     saveP = file('../midData/P.pkl', 'wb')
    #     pickle.dump(self.IOP, saveP, True)
    #     pickle.dump(self.OIP, saveP, True)
    #     saveP.close()
    #
    #     # # 从本地文件中读取字典
    #     # self.M = file('../midData/MUD.pkl', 'rb')
    #     # self.MUO = pickle.load(self.M)
    #     # self.MOU = pickle.load(self.M)
    #     #
    #     # self.R = file('../midData/RUD.pkl', 'rb')
    #     # self.RUO = pickle.load(self.R)
    #     # self.ROU = pickle.load(self.R)
    #     #
    #     # self.Q = file('../midData/QUD.pkl', 'rb')
    #     # self.QUO = pickle.load(self.Q)
    #     # self.QOU = pickle.load(self.Q)
    #     #
    #     # self.IP = file('../midData/P.pkl', 'rb')
    #     # self.PIO = pickle.load(self.IP)
    #     # self.POI = pickle.load(self.IP)

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


    def pickNeighbor(self, user, featureFile):
        # 从本地数据中读取feature()函数中保存的字典
        feature = file(featureFile, 'rb')
        UODict = pickle.load(feature)
        OUDict = pickle.load(feature)

        userOrder =UODict.get(user)
        neighborList = []

        leftIndex = userOrder
        rightIndex = userOrder

        if userOrder >= self.neighborNum/2 and userOrder <= (len(UODict)- self.neighborNum/2):
            while leftIndex > (userOrder - self.neighborNum/2):
                leftIndex -= 1
                neighborID = OUDict.get(leftIndex)
                neighborList.append(neighborID)
            while rightIndex < (userOrder + self.neighborNum/2):
                rightIndex += 1
                neighborID = OUDict.get(rightIndex)
                neighborList.append(neighborID)

        elif userOrder < self.neighborNum/2:
            while leftIndex > 0:
                leftIndex -= 1
                neighborID = OUDict.get(leftIndex)
                neighborList.append(neighborID)
            while rightIndex < self.neighborNum:
                rightIndex += 1
                neighborID = OUDict.get(rightIndex)
                neighborList.append(neighborID)
        else:
            while leftIndex > len(UODict) - self.neighborNum:
                leftIndex -= 1
                neighborID = OUDict.get(leftIndex)
                neighborList.append(neighborID)

            while rightIndex < len(UODict):
                rightIndex += 1
                neighborID = OUDict.get(rightIndex)
                neighborList.append(neighborID)

        # 检测邻居数是否出现异常
        # if len(neighborList) <> neighborNum:
        #     print neighborList, userOrder

        # 从邻居中随机选择一个
        neighbor = choice(neighborList)
        # print neighbor
        return neighbor


    # 构造元路径并随机游走
    def mPath_rWalk(self):
        # U:user, M:MUD, Q:QUD, R:RUD, I:item, P:项目流行度, S:评分
        p1 = 'UMU'
        p2 = 'UQU'
        p3 = 'URU'
        # 找到与用户A评分项目a评分流行度相同的项目b，再找评过项目b的用户B
        p4 = 'UPU'
        # 找到与用户A对产品a评分相同的用户B
        p5 = 'USU'
        mPaths = [p1, p2, p3, p4, p5]

        print 'Generating random meta-path random walks...'
        self.walks = []

        for user in self.dao.user:
            for mp in mPaths:
                if mp == p1:
                    self.walkCount = 10
                if mp == p2:
                    self.walkCount = 10
                if mp == p3:
                    self.walkCount = 10
                if mp == p4:
                    self.walkCount = 10
                if mp == p5:
                    self.walkCount = 10

                for t in range(self.walkCount):
                    path = [(user, 'U')]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'

                    for i in range(self.walkLength / (len(mp)-1)):
                        for tp in mp[1:]:
                            try:
                                if tp == 'M':
                                    nextNode = self.pickNeighbor(lastNode, '../midData/MUD.pkl')
                                if tp == 'Q':
                                    nextNode = self.pickNeighbor(lastNode, '../midData/QUD.pkl')
                                if tp == 'R':
                                    nextNode = self.pickNeighbor(lastNode, '../midData/RUD.pkl')
                                if tp == 'P':
                                    # 从用户评过分的项目中随机选择一个
                                    ratedItem = choice(self.dao.trainingSet_u.get(lastNode).keys())
                                    # 返回相似流行度的项目
                                    samePItem = self.pickNeighbor(lastNode, '../midData/P.pkl')
                                    # 找到评论过这个项目的用户
                                    nextNode = choice(self.dao.trainingSet_i.get(samePItem))
                                if tp == 'S':
                                    # 随机选择用户评过分的项目
                                    item = choice(self.dao.trainingSet_u.get(lastNode).keys())
                                    # 记录这个项目的评分
                                    score = self.dao.trainingSet_u.get(lastNode).get(item)
                                    # 随机选择一个与该项目有相同评分的项目
                                    nextNode = choice(self.ISUDict.get(item).get(score))
                                    #### 如果没有相同评分的情况

                                # if tp == 'U':
                                #     if lastNode == ''









    def initModel(self):
        #feature = self.feature()
        #walk = self.mPath_rWalk()



    # def buildModel(self):
    #     sList = sorted(self.dao.trainingSet_i.iteritems(), key=lambda d: len(d[1]), reverse=True)
    #     print sList


    ###   代码需要优化的内容
    #     1. 没有评分的用户
    #     2. 一个用户的item邻居很少？怎么处理？
