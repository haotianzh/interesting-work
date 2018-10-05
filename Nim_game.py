import numpy as np
import random as rd
class NimState:

    def __init__(self, chips):
        self.chips = chips
        self.lastplayer = 1

    def move(self, m):
        self.chips = self.chips - m
        self.lastplayer = 3 - self.lastplayer

    def isfinish(self):
        canchoose = list(range(1, min(4, self.chips + 1)))
        return canchoose

    def clone(self):
        state = NimState(self.chips)
        state.lastplayer = self.lastplayer
        return state

    def __repr__(self):
        s = "Chips:" + str(self.chips) + " JustPlayed:" + str(self.lastplayer)
        return s
class Node:

    def __init__(self, parentNode, move, state):
        self.parentNode = parentNode
        self.move = move
        self.state = state
        self.childsNode = []
        self.untriedmoves = state.isfinish()
        self.visits = 0
        self.wins = 0
        self.lastplayer = state.lastplayer

    def search(self, state):
        player = state.lastplayer
        chips = state.chips
        if self.state.lastplayer == player and self.state.chips == chips:
            return self.finalselect()
        target = None
        for child in self.childsNode:
            target = child.search(state)
            if target:
                return target
        return target

    def finalselect(self):
        s = sorted(self.childsNode, key=lambda x: x.wins / x.visits)[-1]
        return s

    def UTCSelect(self):
        s = sorted(self.childsNode, key=lambda c: c.wins / (c.visits) + np.sqrt(2 * np.log(self.visits) / (c.visits)))[-1]
        return s

    def addNode(self, m, s):
        childnode = Node(parentNode=self, move=m, state=s)
        self.untriedmoves.remove(m)
        self.childsNode.append(childnode)
        return childnode

    def update(self, result):
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untriedmoves) + "]"

class UTC:

    def __init__(self):
        pass

    def train(self, rootstate, iterations):
        rootnode = Node(parentNode=None, move=None, state=rootstate)
        for i in range(iterations):
            node = rootnode
            state = rootstate.clone()
            # select
            while node.childsNode and not node.untriedmoves:
                node = node.UTCSelect()
                state.move(node.move)

            # expand
            if node.untriedmoves:
                m = rd.choice(node.untriedmoves)
                state.move(m)
                statecopy = state.clone()
                node = node.addNode(m, statecopy)

            # quickly finish match

            while state.isfinish():
                m = rd.choice(state.isfinish())
                state.move(m)

            # update

            while node:
                result = 1 if state.lastplayer == node.lastplayer else 0
                node.update(result)
                node = node.parentNode

        return rootnode

    def search(self, rootstate, iterations):
        rootnode = Node(parentNode=None, move=None, state=rootstate)
        for i in range(iterations):
            node = rootnode
            state = rootstate.clone()
            # select
            while node.childsNode and not node.untriedmoves:
                node = node.UTCSelect()
                state.move(node.move)

            # expand
            if node.untriedmoves:
                m = rd.choice(node.untriedmoves)
                state.move(m)
                node = node.addNode(m, state)

            # quickly finish match

            while state.isfinish():
                m = rd.choice(state.isfinish())
                state.move(m)

            # update


            while node:
                result = 1 if state.lastplayer == node.lastplayer else 0
                node.update(result)
                node = node.parentNode

        return sorted(rootnode.childsNode, key=lambda x: x.visits)[-1].move

def UCTPlay():
    utc = UTC()
    rootstate = NimState(chips=15)
    while rootstate.isfinish():
        print(str(rootstate))
        if rootstate.lastplayer == 1:
            m = utc.search(rootstate, iterations=100)
            player = rootstate.lastplayer
            print(str(player) + "号玩家取走了" + str(m) + "块")
            rootstate.move(m)
        else:
            m = utc.search(rootstate, iterations=1000)
            player = rootstate.lastplayer
            print(str(player) + "号玩家取走了" + str(m) + "块")
            rootstate.move(m)
    print("-" * 15)
    if rootstate.lastplayer == 1:
        print("1号玩家输了！")
    else:
        print("2号玩家输了！")

def trainAndPlay():
    utc = UTC()
    rootstate = NimState(chips=6)
    rootnode = utc.train(rootstate, iterations=50000)
    print("你是2号玩家,游戏开始...")
    while rootstate.isfinish():
        m = rootnode.search(rootstate).move
        print(rootstate)
        rootstate.move(m)
        print("电脑取走了{0}块,还剩{1}块".format(m, rootstate.chips))
        mymove = input("请输入你要取走的数量:")
        mymove = int(mymove)
        rootstate.move(mymove)
        print("玩家取走了{0}块,还剩{1}块".format(mymove, rootstate.chips))
    if rootstate.lastplayer == 1:
        print("游戏结束，电脑输了！")
    else:
        print("游戏结束，玩家输了！")


def playWithComputer():
    utc = UTC()
    rootstate = NimState(chips=15)
    print("你是2号玩家,游戏开始...")
    while rootstate.isfinish():
        m = utc.search(rootstate, iterations=1000)
        rootstate.move(m)
        print("电脑取走了{0}块,还剩{1}块".format(m, rootstate.chips))
        if not rootstate.isfinish():
            break
        mymove = input("请输入你要取走的数量:")
        mymove = int(mymove)
        rootstate.move(mymove)
        print("玩家取走了{0}块,还剩{1}块".format(mymove, rootstate.chips))

    if rootstate.lastplayer == 1:
        print("游戏结束，电脑输了！")
    else:
        print("游戏结束，玩家输了！")

if __name__ == '__main__':
    # UCTPlay()
    playWithComputer()
    # trainAndPlay()


