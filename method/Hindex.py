##coding:utf-8

from baseclass.SSDetection import SSDetection
from data.social import SocialDAO
from tool import config
from tool.file import FileIO
import copy
import math
import numpy as np

class Hindex(SSDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, label=None, relation=list(), fold='[1]'):
        super(Hindex, self).__init__(conf, trainingSet, testSet, label, relation, fold)


    def readConfiguration(self):
        super(Hindex, self).readConfiguration()
        # 注入攻击前的用户关系
        userRelation = FileIO.loadRelationship(self.config, self.config['shillingBefore'])
        self.fao = SocialDAO(self.config, userRelation)

        options = config.LineConfig(self.config['Hindex'])
        self.ratio = float(options['-r'])

    # 计算Hindex
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
    def cacuHOrder(self, data, dataSource, order=float('inf')):
        # data: follower or followee
        # dataSource: 用户数据源，即self.fao.user //攻击前用户关系  或 self.sao.user// 攻击后用户关系
        # order: H-index的阶数, float('inf') // float('inf')表示无穷
        before = -1
        after = 1
        i = -1
        self.hIndexDictList = []
        while before != after:
            # 计算0阶Hindex
            if i == -1:
                self.h_0Dict = {}
                before = 0
                for user in dataSource:
                    #print user, dataSource(user)
                    self.h_0Dict[user] = len(data(user))
                    before += len(data(user))
                #print 'Hindex^ 0 :', self.h_0Dict
                self.hIndexDictList.append(self.h_0Dict)
                i += 1
            # 计算N阶Hindex
            else:
                # h^n
                if i >=1 :
                    before = after
                    self.h_nDict = copy.deepcopy(self.h_iDict)
                self.h_iDict = {}
                after = 0
                for user in dataSource:
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
        if order < i:
            print self.hIndexDictList[order]
            return self.hIndexDictList[order]
        else:
            print self.hIndexDictList[-1]
            return self.hIndexDictList[-1]

    #计算用户核数的变化量
    def cacuChanges(self, shillingData, data, sortStyle):
        # shillingData 攻击后的数据
        # data 攻击前的数据
        hChanges = {}
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
        print sorted(hChanges.items(), key=lambda i: i[1], reverse=sortStyle)
        #print sorted(crossEntropy.items(), key=lambda i: i[1], reverse=True)
        return sorted(hChanges.items(), key=lambda i: i[1], reverse=sortStyle)


    def buildModel(self):
        # print self.sao.relation
        # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        # construction of the linkDict
        #print self.sao.relation
        #print '----------------------------------------'
        #print self.sao.followers
        #print '----------------------------------------'
        # print self.sao.followees
        # print '----------------------------------------'
        #print self.sao.user
        #print self.dao.user
        #print '----------------------------------------'
        #print self.labels

        # followers before shilling attack
        print 'the coreness of user\'s followers before shilling attack'
        followerHindex = self.cacuHOrder(self.fao.getFollowers, self.fao.user, )
        # print sorted(followerHindex.items(), key=lambda i: i[1], reverse=True)

        # followees before shilling attack
        print 'the coreness of user\'s followees before shilling attack'
        followeeHindex = self.cacuHOrder(self.fao.getFollowees, self.fao.user, )
        #print sorted(followerHindex.items(), key=lambda i: i[1], reverse=True)

        # followers after shilling attack
        print 'the coreness of user\'s followers after shilling attack '
        shillingFollowerHindex = self.cacuHOrder(self.sao.getFollowers, self.sao.user, )
        #print sorted(shillingFollowerHindex.items(), key=lambda i: i[1], reverse=True)

        # followees after shilling attack
        print 'the coreness of user\'s followees after shilling attack'
        shillingFolloweeHindex = self.cacuHOrder(self.sao.getFollowees, self.sao.user, )
        #print sorted(shillingFolloweeHindex.items(), key=lambda i: i[1], reverse=True)


        # compare the changes of followers
        print 'the changes of followers'
        self.followerChanges = self.cacuChanges(shillingFollowerHindex, followerHindex, True)

        # compare the changes of followees
        print 'the changes of followees·················································'
        self.followeeChanges = self.cacuChanges(shillingFolloweeHindex, followeeHindex, True)
        #print followerChanges
        #print followeeChanges

        # 建立特征矩阵
        # userID，评分，changesFellowerCoreness，changesFelloweeCoreness


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


# hindex的阶数不同，对检测结果有什么影响