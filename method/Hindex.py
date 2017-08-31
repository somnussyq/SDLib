from baseclass.SSDetection import SSDetection

class Hindex(SSDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, label=None, relation=list(), fold='[1]'):
        super(Hindex, self).__init__(conf, trainingSet, testSet, label, relation, fold)

    def hIndex(self, citations):
        N = len(citations)
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

    def cacuHOrder(self, data):
        before = -1
        after = 1
        i = -1
        whetherCode2 = True
        while before != after:
            # h^0
            if i == -1:
                self.h_0Dict = {}
                before = 0
                for user in self.sao.user:
                    self.h_0Dict[user] = len(data(user))
                    before += len(data(user))
                print 'Hindex^ 0 :',self.h_0Dict
                i += 1
            else:
                # h^n
                if whetherCode2 == False:
                    before = after
                whetherCode2 = False
                self.h_nDict = {}
                after = 0
                for user in self.sao.user:
                    neighborHList = []
                    neighbor = data(user).keys()
                    for tmp in neighbor:
                        neighborHList.append(self.h_0Dict[tmp])
                    #print neighborHList
                    hn = self.hIndex(neighborHList)
                    #print hn
                    self.h_nDict[user] = hn
                    after += hn
                i += 1
                print 'Hindex^', i, ':', self.h_nDict
        return self.h_nDict

    def buildModel(self):
        # construction of the linkDict
        #print self.sao.relation
        #print '----------------------------------------'
        #print self.sao.followers
        #print '----------------------------------------'
        # print self.sao.followees
        # print '----------------------------------------'
        #print self.sao.user
        #print '----------------------------------------'

        # followers
        print 'followers'
        self.cacuHOrder(self.sao.getFollowers)
        print '********************************************************'
        # followees
        print 'followees'
        self.cacuHOrder(self.sao.getFollowees)







