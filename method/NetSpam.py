# coding:UTF-8

from baseclass.SDetection import SDetection
from tool import config
from random import choice,shuffle
import numpy as np
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class NetSpam(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(NetSpam, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(NetSpam, self).readConfiguration()
        options = config.LineConfig(self.config['NetSpam'])
        self.s = int(options['-s'])
        self.b = float(options['-b'])
        self.epoch = int(options['-ep'])
        self.rate = float(options['-r'])
        self.dim = int(options['-dim'])
        self.filter = int(options['-filter'])
        self.negCount = int(options['-negCount'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI, self.regD, self.regE = float(regular['-u']), float(regular['-i']), float(regular['-D']),\
                                                     float(regular['-E'])

    def printAlgorConfig(self):
        super(NetSpam, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['methodName'] + ':'
        print 'Length of levels', self.s
    #     print 'Dimension of user embedding', self.walkDim
    #     print '=' * 80



    # 方法1：用户特征矩阵(DEV,EXT)分解+ Rate 评分矩阵分解
    # 方法2：一阶(二阶、三阶)点互信息分解

# 计算项目及用户特征
    def feature(self):
        # 用户评分偏差
        self.DEV = {}
        # 用户极端评分
        self.EXT = {}


        self.dao.u = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
        self.dao.i = dict(self.dao.trainingSet_i, **self.dao.testSet_i)

        for user in self.dao.u:
            self.EXT[user] = {}
            self.DEV[user] = {}
            for item in self.dao.u[user]:
                rate = self.dao.u[user][item]
                if rate == 1.0 or rate == 5.0:
                    self.EXT[user][item] = 1
                else:
                    self.EXT[user][item] = 0

                dev = abs(self.dao.itemMeans[item] - rate) / 4
                if dev > self.b:
                    self.DEV[user][item] = 1
                else:
                    self.DEV[user][item] = 0

        # print self.EXT
        # print '-----------------'
        # print self.DEV

    def predictRating(self,user,item):
        u = self.dao.all_User[user]
        i = self.dao.all_Item[item]
        return self.P[u].dot(self.Q[i])

    # constructing SPPMI matrix and dict
    def SPMMI(self):
        self.SPPMI = defaultdict(dict)
        D = len(self.dao.user)
        print 'Constructing SPPMI matrix...'
        # for larger data set has many items, the process will be time consuming
        self.occurrence = defaultdict(dict)
        self.occuItem = defaultdict(dict)
        for user1 in self.dao.all_User:
            iList1, rList1 = self.dao.allUserRated(user1)
            # 过滤掉评分数量小于filter的用户user1
            if len(iList1) < self.filter:
                continue
            for user2 in self.dao.all_User:
                if user1 == user2:
                    continue
                if not self.occurrence[user1].has_key(user2):
                    iList2, rList2 = self.dao.allUserRated(user2)
                    # 过滤掉评分数量小于filter的用户user2
                    if len(iList2) < self.filter:
                        continue
                    # count：共同评分数
                    self.occuItem[user1][user2] = []
                    self.occuItem[user1][user2].append(set(iList1).intersection(set(iList2)))
                    count = len(set(iList1).intersection(set(iList2)))
                    # 共同评分数小于filter则为默认值
                    if count > self.filter:
                        self.occurrence[user1][user2] = count
                        self.occurrence[user2][user1] = count

        maxVal = 0
        frequency = {}
        for user1 in self.occurrence:
            frequency[user1] = sum(self.occurrence[user1].values()) * 1.0
        D = sum(frequency.values()) * 1.0
        # maxx = -1
        for user1 in self.occurrence:
            for user2 in self.occurrence[user1]:
                try:
                    val = max([log(self.occurrence[user1][user2] * D / (frequency[user1] * frequency[user2]), 2) - log(
                        self.negCount, 2), 0])
                except ValueError:
                    print self.SPPMI[user1][user2]
                    print self.SPPMI[user1][user2] * D / (frequency[user1] * frequency[user2])
                if val > 0:
                    if maxVal < val:
                        maxVal = val
                    self.SPPMI[user1][user2] = val
                    self.SPPMI[user2][user1] = self.SPPMI[user1][user2]
        print self.occuItem

        # normalize
        for user1 in self.SPPMI:
            for user2 in self.SPPMI[user1]:
                self.SPPMI[user1][user2] = self.SPPMI[user1][user2] / maxVal


    # def SPMMI_2(self):
    #     self.SPPMI_2 = defaultdict(dict)
    #     for user1 in self.dao.all_User:
    #         for user2 in self.dao.all_User:
    #             for user3 in self.dao.all_User:
    #                 set_u1u3 = self.occurrence[user1][user3]
    #                 set_u1u2 = self.




    def mataPath(self):
        pass

    def initModel(self):
        #self.feature()
        self.SPMMI()
        #print self.SPPMI



    def buildModel(self):
        pass
        # self.P = np.random.rand(len(self.dao.all_User) + 1, self.dim) / 10  # latent user matrix
        # self.Q = np.random.rand(len(self.dao.all_Item) + 1, self.dim) / 10  # latent item matrix
        # self.D = np.random.rand(len(self.dao.all_Item) + 1, self.dim) / 10  # dev item embedding
        # self.E = np.random.rand(len(self.dao.all_Item) + 1, self.dim) / 10  # ext item embedding
        # self.Dw = np.random.rand(len(self.dao.all_User) + 1) / 10  # bias value of user in DEV
        # self.Dc = np.random.rand(len(self.dao.all_Item) + 1) / 10  # bias value of item in DEV
        # self.Ew = np.random.rand(len(self.dao.all_User) + 1) / 10  # bias value of user in EXT
        # self.Ec = np.random.rand(len(self.dao.all_Item) + 1) / 10  # bias value of item in EXT
        #
        #
        #
        # iteration = 0
        # while iteration < self.epoch:
        #     self.loss = 0
        #     self.rLoss = 0
        #     self.dLoss = 0
        #     self.eLoss = 0
        #
        #     for user in self.dao.u:
        #         for item in self.dao.u[user]:
        #             rating = self.dao.u[user][item]
        #             error = rating - self.predictRating(user, item)
        #             u = self.dao.all_User[user]
        #             i = self.dao.all_Item[item]
        #             p = self.P[u]
        #             q = self.Q[i]
        #             self.loss += error ** 2
        #             self.rLoss += error ** 2
        #             # update latent vectors
        #             self.P[u] += self.rate * (error * q - self.regU * p)
        #             self.Q[i] += self.rate * (error * p - self.regI * q)
        #
        #     for user in self.DEV:
        #         u = self.dao.all_User[user]
        #         for item in self.DEV[user]:
        #             v = self.dao.all_Item[item]
        #             dev = self.DEV[user][item]
        #             m = self.D[v]
        #             diff = dev - p.dot(m) - self.Dw[v] -self.Dc[u]
        #             self.loss += diff ** 2
        #             self.dLoss += diff ** 2
        #             # update latent vectors
        #             self.P[u] += self.rate * diff * m
        #             self.D[v] += self.rate * diff * p
        #             self.Dw[v] += self.rate * diff
        #             self.Dc[u] += self.rate * diff
        #
        #     self.loss += self.regU *(self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum() \
        #                  + self.regD *(self.D * self.D).sum()
        #
        #     iteration += 1
        #     print 'iteration:', iteration




        # # preparing examples
        # self.training = []
        # self.trainingLabels = []
        # self.test = []
        # self.testLabels = []
        #
        # for user in self.dao.trainingSet_u:
        #     self.training.append(self.P[self.dao.all_User[user]])
        #     self.trainingLabels.append(self.labels[user])
        # for user in self.dao.testSet_u:
        #     self.test.append(self.P[self.dao.all_User[user]])
        #     self.testLabels.append(self.labels[user])



    def predict(self):
       pass
       # classifier = DecisionTreeClassifier(criterion='entropy')
       # classifier.fit(self.training, self.trainingLabels)
       # pred_labels = classifier.predict(self.test)
       # print 'Decision Tree:'
       # return pred_labels