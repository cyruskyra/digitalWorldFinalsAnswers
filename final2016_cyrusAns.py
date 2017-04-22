##### 2016 Finals
##### Wang Nian Yu. Cyrus
##### 1002176
##### 16F09

import better_exceptions

##### Question 3

def maxOccurrences(inp):
	inpList = map(int,inp.split(' '))
	occurrenceDict = {}
	for i in inpList:
		if occurrenceDict.has_key(i):
			occurrenceDict[i] += 1
		else:
			occurrenceDict[i] = 1
	allOccurrences = occurrenceDict.values()
	maxOccurrences = max(allOccurrences)
	occurrenceList = []
	for i in occurrenceDict:
		if occurrenceDict[i] == maxOccurrences:
			occurrenceList.append(i)
	return (sorted(occurrenceList),maxOccurrences)

print 'Question 3 Test Cases \n'

print 'test 1'
inp = '2 3 40 3 5 4 -3 3 3 2 0'
print maxOccurrences(inp)

print 'test 2'
inp = '9 30 3 9 3 2 4'
print maxOccurrences(inp)

print '\n'

##### Question 4

from libdw import sm

class RingCounter(sm.SM):
	startState = 0
	
	def getNextValues(self,state,inp):
		threeBit = {0:'000',1:'001',2:'010',3:'011',4:'100',5:'101',6:'110',7:'111'}
		if inp == 1:
			return(0,'000')
		else:
			state += 1
			if state == 8:
				state = 0
				return(0,'000')
			else:
				return(state,threeBit[state])

print 'Question 4 Test Cases \n'

print 'test 1'
r = RingCounter()
print r.transduce([0,0,0,0,0,0,0,0,0])

print 'test 2'
r = RingCounter()
print r.transduce([0,0,0,1,0,0,0,0,0])

print 'test 3'
r = RingCounter()
print r.transduce([0,0,0,1,0,0,1,0,0])

print '\n'

##### Question 5

class Avatar:
	def __init__(self,name,hp = 100,position = (1,1)):
		self.name = name
		self.hp = hp
		self.position = position

	def getName(self):
		return self.name
	def setName(self,inp):
		self.name = inp

	def getHP(self):
		return self.hp
	def setHP(self,inp):
		self.hp = inp

	def getPosition(self):
		return self.position
	def setPosition(self,inp):
		self.position = inp

	def heal(self,inp = 1):
		if inp > 0:
			self.hp += inp

	def attacked(self,inp = -1):
		if inp < 0:
			self.hp += inp

print 'Question 5 Test Cases \n'

print 'test 1: __init__'
a = Avatar('John')
print (a.name,a.hp,a.position)

print 'test 1: __init__'
a = Avatar('Jane',150,(10,10))
print (a.name,a.hp,a.position)

print 'test 3: getName(), setName()'
a = Avatar('John')
a.setName('Jude')
print a.getName()

print 'test 4: getPosition(), setPosition()'
a = Avatar('John')
a.setPosition((1,3))
print a.getPosition()

print 'test 5: getHP(), setHP()'
a = Avatar('John')
a.setHP(200)
print a.getHP()

print 'test 6: heal()'
a = Avatar('John')
a.heal(5)
print a.getHP()

print 'test 7: attacked()'
a = Avatar('John')
a.attacked(20)
print a.getHP()

print 'test 8: heal()'
a = Avatar('John')
a.heal()
print a.getHP()

print 'test 9: attacked()'
a = Avatar('John')
a.attacked()
print a.getHP()

print 'test 10: heal(), attacked() '
a = Avatar('John')
a.attacked(2)
a.heal(-2)
print a.getHP()

print '\n'

##### Question 6

from copy import deepcopy

class Map(object):
	def __init__(self,world):
		self.world = deepcopy(world)

	def whatIsAt(self,coord):
		if self.world.has_key(coord):
			info = self.world[coord]
			if info == 'x':
				return 'Exit'
			elif info == 0:
				return 'Wall'
			elif info > 0:
				return 'Food'
			elif info < 0:
				return 'Enemy'
		else:
			return 'Empty'

	def getEnemyAttack(self,coord):
		if self.whatIsAt(coord) == 'Enemy':
			return self.world[coord]
		else:
			return False

	def getFoodEnergy(self,coord):
		if self.whatIsAt(coord) == 'Food':
			return self.world[coord]
		else:
			return False

	def removeEnemy(self,coord):
		if self.whatIsAt(coord) == 'Enemy':
			del self.world[coord]
			return True
		else:
			return False

	def eatFood(self,coord):
		if self.whatIsAt(coord) == 'Food':
			del self.world[coord]
			return True
		else:
			return False

	def getExitPosition(self):
		for i in self.world:
			if self.world[i] == 'x':
				return i
		return None

print 'Question 6 Test Cases \n'

world={(0,0):0, (1,0):0 , (2,0):0, (3,0): 0, (4,0):0, (5,0): 0,
(0,1):0, (1,1): 2, (2,1):-3, (5,1): 0, (0,2):0, (5,2): 0, (0,3):0,
(5,3): 0, (0,4):0, (5,4): 0, (0,5):0, (1,5):0 , (2,5):0, (3,5): 0,
(4,5):'x', (5,5): 0}

print 'test 1: object instantiation'
m = Map(world)
print m.world

print 'test 2: whatIsAt'
print m.whatIsAt((1,0))

print 'test 3: whatIsAt'
print m.whatIsAt((2,1))

print 'test 4: whatIsAt'
print m.whatIsAt((1,1))

print 'test 5: getFoodEnergy'
w1 = m.getFoodEnergy((1,1))
w2 = m.getFoodEnergy((3,3))
print (w1,w2)

print 'test 6: getEnemyAttack'
w1 = m.getEnemyAttack((2,1))
w2 = m.getEnemyAttack((3,3))
print (w1,w2)

print 'test 7: removeEnemy'
w1 = m.getEnemyAttack((2,1))
w2 = m.removeEnemy((2,1))
w3 = m.getEnemyAttack((2,1))
print (w1,w2,w3)

print 'test 8: whatIsAt'
print m.whatIsAt((1,4))

print 'test 9: getFoodEnergy'
print m.getFoodEnergy((1,4))

print 'test 10: getEnemyAttack'
print m.getEnemyAttack((1,4))

print 'test 11: whatIsAt'
print m.whatIsAt((4,5))

print 'test 12: getExitPosition'
print m.getExitPosition()

print 'test 13: eatFood'
w1 = m.whatIsAt((1,1))
w2 = m.eatFood((1,1))
w3 = m.whatIsAt((1,1))
print (w1,w2,w3)

print 'test 14: test aliasing'
print m.world == world

print '\n'

##### Question 7

from operator import add

class DW2Game(sm.SM):
	def __init__(self,Avatar,Map):
		self.startState = (deepcopy(Avatar),deepcopy(Map))

	def getNextValues(self,state,inp):
		nextState = deepcopy(state)

		if inp[1] == 'up':
			newPosition = tuple(map(add,deepcopy(nextState[0].position),(0,-1)))
		elif inp[1] == 'down':
			newPosition = tuple(map(add,deepcopy(nextState[0].position),(0,1)))
		elif inp[1] == 'left':
			newPosition = tuple(map(add,deepcopy(nextState[0].position),(-1,0)))
		elif inp[1] == 'right':
			newPosition = tuple(map(add,deepcopy(nextState[0].position),(1,0)))

		if inp[0] == 'move':
			if nextState[1].whatIsAt(newPosition) in ['Empty','Food','Exit']:
				nextState[0].position = newPosition
				if nextState[1].whatIsAt(newPosition) == 'Food':
					nextState[0].heal(nextState[1].getFoodEnergy(newPosition))
					nextState[1].eatFood(newPosition)
			elif nextState[1].whatIsAt(newPosition) == 'Enemy':
				nextState[0].attacked(nextState[1].getEnemyAttack(newPosition))
		elif inp[0] == 'attack':
			if nextState[1].whatIsAt(newPosition) == 'Enemy':
				nextState[1].removeEnemy(newPosition)

		return (nextState[0],nextState[1]),(nextState[0].name,nextState[0].position,nextState[0].hp)

	def done(self,state):
		if state[1].whatIsAt(state[0].position) == 'Exit':
			return True
		else:
			return False

print 'Question 7 Test Cases \n'

world2={(0,0):0, (1,0):0 , (2,0):0, (3,0): 0, (4,0):0, (5,0): 0, (0,1):0, (5,1): 0, (0,2):0, (1,2): -2, (5,2): 0, (0,3):0, (2,3): 3, (5,3): 0, (0,4):0, (5,4): 0, (0,5):0, (1,5):0, (2,5):0, (3,5): 0, (4,5):'x', (5,5): 0,}

print 'test 1'
inp = [('move','down'),('attack','down'),('move','down'),('move','down'),('move','down'),('move','right'),('move','right'),('move','right'),('move','down'),('move','down')]
av = Avatar('John')
m = Map(world2)
g = DW2Game(av,m)
print g.transduce(inp)

print 'test 2'
inp=[('move','left'),('move','right'),('move','right'),('move','right'),('move','right'),('move','down'),('move','down'),('move','down'),('move','up')]
av=Avatar('John')
m=Map(world2)
g=DW2Game(av,m)
print g.transduce(inp)

print 'test 3'
inp=[('move','right'),('move','right'),('move','right'),('move','down'),('move','left'),('move','left'),('move','left'),('attack','left'),('move','left')]
av=Avatar('John')
m=Map(world2)
g=DW2Game(av,m)
print g.transduce(inp)

print 'test 4'
inp=[('move','right'),('move','right'),('move','right'),('move','down'),('move','left'),('move','left'),('move','left'),('attack','left'),('move','left'),('move','left'),('move','down'),('move','right')]
av=Avatar('John')
m=Map(world2)
g=DW2Game(av,m)
print g.transduce(inp)

print 'test 5'
inp=[('move','right'),('move','right'),('move','right'),('move','down'),('move','left'),('move','left'),('move','left'),('attack','left'),('move','left'),('move','left'),('move','down'),('move','right'),('move','right'),('move','right'),('move','down'),('move','down'),('move','down')]
av=Avatar('John')
m=Map(world2)
g=DW2Game(av,m)
print g.transduce(inp)

print 'test 6'
av=Avatar('John')
m=Map(world2)
g=DW2Game(av,m)
g.start()
n,o=g.getNextValues(g.startState,('move','right'))
ans = g.state[0].getPosition() == n[0].getPosition()
print ans, g.state[0].getPosition(), n[0].getPosition()

print '\n'

##### Question 8

class Map(object):
	def __init__(self,world):
		self.world = deepcopy(world)

	def whatIsAt(self,coord):
		if self.world.has_key(coord):
			info = self.world[coord]
			if info == 'x':
				return 'Exit'
			elif info == 0:
				return 'Wall'
			elif info > 0:
				return 'Food'
			elif info < 0:
				return 'Enemy'
		else:
			return 'Empty'

	def getEnemyAttack(self,coord):
		if self.whatIsAt(coord) == 'Enemy':
			return self.world[coord]
		else:
			return False

	def getFoodEnergy(self,coord):
		if self.whatIsAt(coord) == 'Food':
			return self.world[coord]
		else:
			return False

	def removeEnemy(self,coord):
		if self.whatIsAt(coord) == 'Enemy':
			del self.world[coord]
			return True
		else:
			return False

	def eatFood(self,coord):
		if self.whatIsAt(coord) == 'Food':
			del self.world[coord]
			return True
		else:
			return False

	def getExitPosition(self):
		for i in self.world:
			if self.world[i] == 'x':
				return i
		return None

	def getSearchMap(self):
		world = deepcopy(self.world)

		coordX = []
		coordY = []
		for i in world:
			coordX.append(i[0])
			coordY.append(i[1])
		minX = min(coordX)
		maxX = max(coordX)
		minY = min(coordY)
		maxY = max(coordY)

		valueList = (world.values())
		valueList.remove('x')
		offset = max(valueList)

		allCoords = []
		for x in xrange(minX,maxX+1):
			for y in xrange(minY,maxY+1):
				allCoords.append((x,y))

		finalDict = {}
		for i in allCoords:
			directionList = [tuple(map(add,i,(0,-1))),tuple(map(add,i,(1,0))),tuple(map(add,i,(0,1))),tuple(map(add,i,(-1,0)))]
			
			directionCount = 0
			positionDict = {}
			for j in directionList:
				if self.world.has_key(j) and self.world[j] not in ['x',0]:
					positionDict[directionCount] = (j,offset - self.world[j])
				elif self.world.has_key(j) and self.world[j] == 0:
					positionDict[directionCount] = (j,1000)
				else:
					positionDict[directionCount] = (j,offset)
				directionCount += 1
			
			validPositionDict = deepcopy(positionDict)
			for j in positionDict:
				if positionDict[j][0][0] < minX or positionDict[j][0][0] > maxX or positionDict[j][0][1] < minY or positionDict[j][0][1] > maxY:
					del validPositionDict[j]
			finalDict[i] = validPositionDict

		return finalDict

print 'Question 8 Test Cases \n'

world = {(0,0):0, (1,0):0 , (2,0):0, (0,1):0, (1,1):-2, (2,1): 0, (0,2):0, (1,2): 'x', (2,2): 0}
print 'test 1'
m = Map(world)
print m.getSearchMap()

world = {(0,0):0, (1,0):0 , (2,0):0, (0,1):0, (1,1):3, (2,1): 0, (0,2):0, (1,2): 'x', (2,2): 0}
print 'test 2'
m = Map(world)
print m.getSearchMap()

world = {(0,0):0, (1,0):0 , (2,0):0, (3,0): 0, (4,0):0, (5,0): 0, (0,1):0, (5,1): 0, (0,2):0, (1,2): -2, (5,2): 0, (0,3):0, (2,3): 3,(5,3): 0, (0,4):0, (5,4): 0, (0,5):0, (1,5):0 , (2,5):0, (3,5): 0, (4,5):'x', (5,5): 0}
print 'test 3'
m = Map(world)
print m.getSearchMap()

print '\n'

##### Question 9

"""
Procedures and classes for doing uniform cost search, always with
dynamic programming.  Becomes A* if a heuristic is specified. 
"""

from libdw import util

somewhatVerbose = False
"""If ``True``, prints a trace of the search"""
verbose = False
"""If ``True``, prints a verbose trace of the search"""

class SearchNode:
    """A node in a search tree"""
    def __init__(self, action, state, parent, actionCost):
        self.state = state
        self.action = action
        """Action that moves from ``parent`` to ``state``"""
        self.parent = parent
        if self.parent:
            self.cost = self.parent.cost + actionCost
            """The cost of the path from the root to ``self.state``"""
        else:
            self.cost = actionCost
        
    def path(self):
        """:returns: list of ``(action, state)`` pairs from root to this node"""
        if self.parent == None:
            return [(self.action, self.state)]
        else:
            return self.parent.path() + [(self.action, self.state)]

    def inPath(self, s):
        """
        :returns: ``True`` if state ``s`` is in the path from here to
         the root
        """
        if s == self.state:
            return True
        elif self.parent == None:
            return False
        else:
            return self.parent.inPath(s)

    def __repr__(self):
        if self.parent == None:
            return str(self.state)
        else:
            return repr(self.parent) + \
                   "-"+str(self.action)+"->"+str(self.state)

    __str__ = __repr__

class PQ:
    """
    Slow implementation of a priority queue that just finds the
    minimum element for each extraction.
    """
    def __init__(self):
        """Create a new empty priority queue."""
        self.data = []
    def push(self, item, cost):
        """Push an item onto the priority queue.
           Assumes items are instances with an attribute ``cost``."""
        self.data.append((cost, item))
    def pop(self):
        """Returns and removes the least cost item.
           Assumes items are instances with an attribute ``cost``."""
        (index, cost) = util.argmaxIndex(self.data, lambda (c, x): -c)
        return self.data.pop(index)[1] # just the data item
    def isEmpty(self):
        """Returns ``True`` if the PQ is empty and ``False`` otherwise."""
        return len(self.data) == 0
    def __str__(self):
        return 'PQ('+str(self.data)+')'
        

def search(initialState, goalTest, actions, successor,
           heuristic = lambda s: 0, maxNodes = 10000):
    """
    :param initialState: root of the search
    :param goalTest: function from state to Boolean
    :param actions: a list of possible actions
    :param successor: function from state and action to next state and cost
    :param heuristic: function from state to estimated cost to reach a goal;
        defaults to a heuristic of 0, making this uniform cost search
    :param maxNodes: kill the search after it expands this many nodes
    :returns: path from initial state to a goal state as a list of
           (action, state) tuples
    """
    startNode = SearchNode(None, initialState, None, 0)
    if goalTest(initialState):
        return startNode.path()
    agenda = PQ()
    agenda.push(startNode, 0)
    expanded = {}
    count = 1
    while (not agenda.isEmpty()) and maxNodes > count:
        if verbose: print "agenda: ", agenda
        n = agenda.pop()
        if not expanded.has_key(n.state):
            expanded[n.state] = True
            if goalTest(n.state):
                # We're done!
                return n.path(),n.cost
            if somewhatVerbose or verbose:
                print "   ", n.cost, ":   expanding: ",  n
            for a in actions:
                (newS, cost) = successor(n.state, a)
                if not expanded.has_key(newS):
                    # We don't know the best path to this state yet
                    count += 1
                    newN = SearchNode(a, newS, n, cost)
                    agenda.push(newN, newN.cost + heuristic(newS))
    print "Search failed after visiting ", count, " states."
    return None

def findPath(Avatar,Map):
	searchMap = Map.getSearchMap()
	initialState = Avatar.position
	goal = Map.getExitPosition()
	actions = [0,1,2,3]

	def goalTest(initialState):
		return goal == initialState

	def successor(state,action):
		if searchMap[state].has_key(action):
			return searchMap[state][action]
		else:
			return (None,100000000000000)

	path = search(initialState,goalTest,actions,successor)
	if path[1] < 1000:
		return path
	else:
		return None

print 'Question 9 Test Cases \n'

world = {(0,0):0, (1,0):0, (2,0):0, (3,0): 0, (4,0):0,(5,0): 0, (0,1):0, (5,1): 0,(0,2):0, (1,2): -2, (5,2): 0, (0,3):0, (2,3): 3, (5,3): 0,(0,4):0, (5,4): 0,(0,5):0, (1,5):0, (2,5):0, (3,5): 0, (4,5):'x', (5,5): 0}
print 'test 1'
av = Avatar('John',position=(1,3))
m = Map(world)
print findPath(av,m)

world = {(0,0):0, (1,0):0, (2,0):0, (3,0): 0, (4,0):0,(5,0): 0, (0,1):0, (5,1): 0, (0,2):0, (1,2): -2, (5,2): 0,(0,3):0, (2,3): 3, (3,3):0, (5,3): 0, (0,4):0, (3,4):0,(5,4): 0, (0,5):0, (1,5):0, (2,5):0, (3,5): 0, (4,5):'x',(5,5): 0}
print 'test 2'
av = Avatar('John',position=(1,3))
m = Map(world)
print findPath(av,m)

world = {(0,0):0, (1,0):0 , (2,0):0, (3,0): 0, (4,0):0,(5,0): 0,(0,1):0, (3,1):0, (5,1): 0,(0,2):0, (1,2): -2,(3,2):0, (5,2): 0,(0,3):0, (2,3): 3, (3,3):0, (5,3):0,(0,4):0, (3,4):0, (5,4): 0,(0,5):0, (1,5):0 , (2,5):0,(3,5): 0, (4,5):'x', (5,5): 0}
print 'test 3'
av = Avatar('John',position=(1,3))
m = Map(world)
print findPath(av,m)