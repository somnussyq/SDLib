##coding:utf-8

from baseclass.SSDetection import SSDetection
from data.social import SocialDAO
from tool import config
from tool.file import FileIO
import copy
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Hindex(SSDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, label=None, relation=list(), fold='[1]'):
        super(Hindex, self).__init__(conf, trainingSet, testSet, label, relation, fold)


    def readConfiguration(self):
        super(Hindex, self).readConfiguration()
        # # 注入攻击前的用户关系
        # userRelation = FileIO.loadRelationship(self.config, self.config['shillingBefore'])
        # self.fao = SocialDAO(self.config, userRelation)

        options = config.LineConfig(self.config['Hindex'])
        self.ratio = float(options['-r'])



    # 数据说明
    #self.fao: 原数据集用户关系 // 真实环境中没有该数据集
    #self.sao: 攻击之后用户关系
    #self.dao.user: 训练集{用户id：用户序号}
    #self.dao.all_user: 训练+测试集{用户id：用户序号}

    # 计算k-truss
    def edgeKtruss(self):
        #self.Ktruss ==> {user1：{user2:triangleCount, user3:triangleCount }}
        self.Ktruss = copy.deepcopy(self.sao.undirect)
        #print self.Ktruss

        # 节点的邻居用集合保存
        self.ktrussSet = {}
        for user, neighbor in self.Ktruss.items():
            if not self.ktrussSet.has_key(user):
                self.ktrussSet[user] = set(neighbor.keys())

        for user, neighbor in self.Ktruss.items():
            for friend, weight in neighbor.items():
                self.Ktruss[user][friend]= len(self.ktrussSet[user] & self.ktrussSet[friend])
        #print self.Ktruss

        # 计算节点的Ktruss： 与之相连边所在三角形的平均值
        self.userKtruss = {}
        for user1 in self.dao.all_User:
            tot = 0
            try:
                for user2, weight in self.Ktruss[user1].items():
                    tot += weight
                count = len(self.Ktruss[user1])
                # round(a,4) ==> 对a保留4位小数
                self.userKtruss[user1] = round(tot*1.0/count,4)
            except(KeyError):
                self.userKtruss[user1] = 0
        #print self.userKtruss
        return self.userKtruss



    # 计算Hindex
    # citations 在Hindex里表示引用关系
    # 在此处表示用户的邻居
    def hIndex(self, citations):
        #print citations
        N = len(citations)
        #print N
        cnts = [0] * (N + 1)
        for i in citations:
            cnts[[i, N][i > N]] += 1
            # print '*************'
            # print cnts
        sum = 0
        for h in range(N, 0, -1):
            if sum + cnts[h] >= h:
                return h
            sum += cnts[h]
        return 0

    # 计算N阶Hindex
    # 高阶Hindex收敛时即为核数 coreness
    # 返回用户的Coreness
    def cacuHOrder(self, data, order):
        # data: 用字典存储用户的邻居信息：follower or followee // self.fao.getFollowers、self.fao.getFollowees
        # order: H-index的阶数:N~float('inf') // float('inf')表示无穷
        before = -1
        after = 1
        i = -1
        self.hIndexDictList = []
        while before != after:
            # 计算0阶Hindex
            if i == -1:
                self.h_0Dict = {}
                before = 0
                for user in self.dao.all_User:
                    self.h_0Dict[user] = len(data(user))
                    before += len(data(user))
                #print 'Hindex^ 0 :', self.h_0Dict
                self.hIndexDictList.append(self.h_0Dict)
                #print self.hIndexDictList
                i += 1

            # 计算N阶Hindex
            else:
                # h^n
                if i >=1 :
                    before = after
                    self.h_nDict = copy.deepcopy(self.h_iDict)
                self.h_iDict = {}
                after = 0
                for user in self.dao.all_User:
                    neighborHList = []
                    neighbor = data(user).keys()
                    for tmp in neighbor:
                        if i == 0:
                            neighborHList.append(self.h_0Dict[tmp])
                        else:
                            neighborHList.append(self.h_nDict[tmp])
                    #print neighborHList
                    hn = self.hIndex(neighborHList)
                    #print hn
                    self.h_iDict[user] = hn
                    after += hn
                i += 1
                #print 'Hindex^', i, ':', self.h_iDict
                self.hIndexDictList.append(self.h_iDict)

        # 返回用户核数字典
        if order < i:
            #print self.hIndexDictList[order]
            return self.hIndexDictList[order]
        else:
            #print self.hIndexDictList[-1]
            return self.hIndexDictList[-1]

    #计算用户核数的变化量
    def cacuChanges(self, shillingData, data):
        # shillingData 攻击后的数据
        # data 攻击前的数据
        # sortType 排序方式 // 根据是否需要排序返回加此参数
        hChanges = {}
        # 计算交叉熵
        #crossEntropy = {}
        for i in shillingData:
            #print i
            after = shillingData[i]
            after = shillingData[i]
            #print after
            before = data.get(i, 0)
            #print before
            change = after - before
            #print after, type(after)
            #if after <= 0:
            #    after = 0.5
            #cEntropy = before * math.log(after)
            #crossEntropy[i] = cEntropy
            #print change
            #print '---------------------------'
            hChanges[i] = change
        #print hChanges
        return hChanges
        # print sorted(hChanges.items(), key=lambda i: i[1] reverse=sortType)
        # #print sorted(crossEntropy.items(), key=lambda i: i[1], reverse=True)
        # return sorted(hChanges.items(), key=lambda i: i[1] reverse=sortType)


    def buildModel(self):
        print 'caculating the k-truss...'
        self.edgeKtruss()
        #print 'trainingSet'
        # followers before shilling attack
        #print 'the coreness of user\'s followers before shilling attack'
        #followerHindex0 = self.cacuHOrder(self.fao.getFollowers,float('inf'))
        #followerHindex = self.cacuHOrder(self.fao.getFollowers, 0)
        # self.followerChanges = self.cacuChanges(followerHindex0, followerHindex)
        # print self.followerChanges


        # followees before shilling attack
        #print 'the coreness of user\'s followees before shilling attack'
        #followeeHindex = self.cacuHOrder(self.fao.getFollowees, 0)
        #print followeeHindex

        # # followers after shilling attack
        print 'the coreness of user\'s followers after shilling attack '
        shillingFollowerDegree = self.cacuHOrder(self.sao.getFollowers, 0)
        shillingFollowerHindex = self.cacuHOrder(self.sao.getFollowers, 1)
        shillingFollowerCoreness = self.cacuHOrder(self.sao.getFollowers, float('inf'))

        # # followees after shilling attack
        #print 'the coreness of user\'s followees after shilling attack'
        shillingFolloweeDegree = self.cacuHOrder(self.sao.getFollowees, 0)
        shillingFolloweeHindex = self.cacuHOrder(self.sao.getFollowees, 1)
        shillingFolloweeCoreness = self.cacuHOrder(self.sao.getFollowees, float('inf'))
        # #print sorted(shillingFolloweeHindex.items(), key=lambda i: i[1], reverse=True)

        # # compare the changes of followers
        #print 'the changes of followers'
        #self.followerChanges = self.cacuChanges(shillingFollowerHindex, followerHindex)
        #print self.followerChanges

        # # compare the changes of followees
        #print 'the changes of followees '
        #self.followeeChanges = self.cacuChanges(shillingFolloweeHindex, followeeHindex)
        #print self.followeeChanges
        # #print followerChanges
        # #print followeeChanges

        # 建立特征矩阵
        # userID，评分，changesFellowerCoreness，changesFelloweeCoreness
        #for user in self.dao.trainingSet_u:


        # print len(self.fao.user)
        # print len(self.dao.user)
        # print len(self.sao.user)

        # preparing examples
        for user in self.dao.trainingSet_u:
            # self.training.append([followerHindex[user], followeeHindex[user],
            #                       shillingFollowerHindex[user], shillingFolloweeHindex[user],
            #                       self.followerChanges[user], self.followeeChanges[user]])
            self.training.append([shillingFollowerDegree[user], shillingFollowerHindex[user],
                                  shillingFollowerCoreness[user],shillingFolloweeDegree[user],
                                  shillingFolloweeHindex[user],shillingFolloweeCoreness[user],
                                  self.userKtruss[user]])
            self.trainingLabels.append(self.labels[user])

        for user in self.dao.testSet_u:
            # self.test.append([followerHindex[user], followeeHindex[user],
            #                       shillingFollowerHindex[user], shillingFolloweeHindex[user],
            #                       self.followerChanges[user], self.followeeChanges[user]])
            self.test.append([shillingFollowerDegree[user], shillingFollowerHindex[user],
                                  shillingFollowerCoreness[user], shillingFolloweeDegree[user],
                                  shillingFolloweeHindex[user], shillingFolloweeCoreness[user],
                              self.userKtruss[user]])
            self.testLabels.append(self.labels[user])


    def predict(self):
        # ##无监督方法##
        #
        # # 正常用户的入度有变化（followee）
        # # 虚假用户的出度变化较大（follower）
        # self.predLabels = np.zeros(len(self.labels))
        # spamList = []
        #
        # # 虚假用户是followee与follower共同变化量最大的N个
        # #虚假用户是followee或follower变化量最大的N个
        # i = 0
        # while i < self.ratio * len(self.labels):
        #     spam = self.followeeChanges[i][0]
        #     self.predLabels[self.sao.user.get(spam)] = 1
        #     i += 1
        # print len(self.predLabels), sum(self.predLabels), self.predLabels,
        #
        # # trueLabels
        # tot = 0
        # for tmp in self.labels:
        #     tot += int(self.labels.get(tmp))
        # print tot
        # #self.testLabels = np.zeros(len(self.labels))
        # self.testLabels = []
        # for j in range(0, len(self.labels)):
        #     self.testLabels.append(0)
        # for i in self.labels:
        #     # print i
        #     # print self.sao.user.get(i, len(self.sao.user)+1)
        #     #print int(self.labels.get(i))
        #     # self.testLabels[int(i) - 1] = self.labels.get(i)
        #     label = int(self.labels.get(i))
        #     self.testLabels[self.sao.user.get(i, len(self.sao.user)+1)] = label
        # print len(self.testLabels), sum(self.testLabels), self.testLabels
        # return self.predLabels

        # 有监督的方法
        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print 'Decision Tree:'
        return pred_labels
        #
        # clfRF = RandomForestClassifier(n_estimators=10)
        # clfRF.fit(self.training, self.trainingLabels)
        # pred_labels = clfRF.predict(self.test)
        # return pred_labels



# hindex的阶数不同，对检测结果有什么影响