import numpy as np

number = [1, 3, 12, 60, 360, 1000]
allcount = sum(number)


class Civilization(object):
    def __init__(self, number, civil_val, policy, location, dis):
        if number % 100000 == 0:
            print('Constructing Civil No.{} with civilVal {}'.format(
                number, civil_val))
        self.no = number
        self.civilVal = civil_val
        self.policy = policy  # policy True == battle
        self.location = location
        self.dis = dis

    def setval(self, val):
        self.civilVal = val

    def develop(self):
        self.civilVal *= 1.01


class Universe(object):
    def __init__(self, R):
        self.civilList = []
        self.civilCount = 0
        self.R = R

    def getCivilList(self):
        return self.civilList

    def add2civilList(self, civil):
        self.civilList.append(civil)
        self.civilCount += 1

    def clear(self):
        cpycivilList = self.civilList[:]
        for i in self.civilList:
            if i.civilVal == 0:
                cpycivilList.remove(i)
        self.civilList = cpycivilList
        return len(cpycivilList)

    def nextEpoch(self, counts):
        label = [0 for i in range(len(self.civilList))]
        i = 0
        while True:
            if i == len(self.civilList) - 1:
                break

            if label[i] == 1 or self.civilList[i].civilVal == 0:
                i += 1
                continue
            else:
                label[i] = 1
                while True:
                    num = np.random.randint(0, len(self.civilList))
                    if label[num] == 0 and self.civilList[i].civilVal != 0:
                        break
                    if 0 not in label[i:]:
                        break
            label[num] = 1
            distance = 0
            dx = self.civilList[i].location[0] - \
                self.civilList[num].location[0]
            dy = self.civilList[i].location[1] - \
                self.civilList[num].location[1]
            dz = self.civilList[i].location[2] - \
                self.civilList[num].location[2]
            distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            if ((self.civilList[i].policy == True) or (self.civilList[num].policy == True)) and (distance <= max(self.civilList[i].dis, self.civilList[num].dis)):
                if self.civilList[i].civilVal >= self.civilList[num].civilVal:
                    self.civilList[i].civilVal -= self.civilList[num].civilVal
                    self.civilList[num].civilVal = 0
                else:
                    self.civilList[num].civilVal -= self.civilList[i].civilVal
                    self.civilList[i].civilVal = 0
            elif distance <= max(self.civilList[i].dis, self.civilList[num].dis):
                if self.civilList[i].civilVal <= self.civilList[num].civilVal:
                    self.civilList[i].civilVal *= 1.2
                else:
                    self.civilList[num].civilVal *= 1.2
            else:
                pass
            i += 1
        if counts % 1 == 0:
            print('Time Epoch Ends')
            for i in self.civilList:
                i.develop()
            count = 0
            for i in self.civilList:
                if i.civilVal == 0:
                    count += 1
            print('{} civils extinct.'.format(count))
            count = 0
            for i in self.civilList:
                if (i.policy == True) and (i.civilVal != 0):
                    count += 1
                    # print(i.civilVal)
            print('{} living civils decide to battle'.format(count))
            count = 0
            for i in self.civilList:
                if (i.policy == False) and (i.civilVal != 0):
                    count += 1
                    # print(i.civilVal)
            print('{} living civils decide to be peace'.format(count))
        remain = self.clear()
        if counts % 1 == 0:
            print("[{}] {} out of {} civils remain".format(
                counts, remain, allcount))
            print('#####################')


# Define civil numbers

val = [100000, 10000, 1000, 100, 10, 1]
peace = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
space = [0.8**(i) for i in range(6)]
unvs = Universe(1)
cnt = 0
policy = False
for i in range(len(number)):
    for j in range(number[i]):
        cnt += 1
        z = np.random.uniform(0, 1)
        if z <= peace[i]:
            policy = False
        else:
            policy = True
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        z = np.random.uniform(0, 1)
        civil = Civilization(cnt, val[i], policy, [x, y, z], space[i])
        unvs.add2civilList(civil)
civillst = unvs.getCivilList()
print(len(civillst))
cnt = 0
timeEpoch = 0
for i in range(200):
    unvs.nextEpoch(i)
exit()
while True:
    pass
