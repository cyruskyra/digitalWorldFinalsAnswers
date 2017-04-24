##### 2015 Finals
##### Wang Nian Yu. Cyrus
##### 1002176
##### 16F09

import better_exceptions

##### Question 3

def compTrace(A):
	n = len(A[0])
	ijList = []
	for i in xrange(n):
		ijList.append(A[i][i])
	return sum(ijList)

print 'Question 3 Test Cases\n'

A = [[2.2,2,3.1],[4,5,6],[7,8,9]]
print compTrace(A)

print '\n'

##### Question 4

def findKey(dInput,strInput):
	keysFound = []
	for i in dInput:
		if dInput[i] == strInput:
			keysFound.append(i)
	return sorted(keysFound)

print 'Question 4 Test Cases\n'

dInput = {1:'singapore', 20:'china', 4:'japan', 5:'china', 10:'japan'}
print findKey(dInput, 'china')
print findKey(dInput, 'korea')

print '\n'

##### Question 5

class Square:
	def __init__(self,x = 0,y = 0,sideLength = 1.0):
		self.x = x
		self.y = y
		self.sideLength = sideLength

	def getCenter(self):
		return (self.x,self.y)

	def getSideLength(self):
		return self.sideLength

	def getArea(self):
		return self.sideLength ** 2

	def getPerimeter(self):
		return self.sideLength * 4

	def containPoint(self,px,py):
		halfLength = self.sideLength/2
		xBoundaries = (self.x - halfLength,self.x + halfLength)
		yBoundaries = (self.y - halfLength,self.y + halfLength)
		return (xBoundaries[0] <= px <= xBoundaries[1]) and (yBoundaries[0] <= py <= yBoundaries[1])

	def containSquare(self,inSquare):
		inSquareHalf = inSquare.sideLength/2
		inSquareX = [inSquare.x + inSquareHalf,inSquare.x - inSquareHalf]
		inSquareY = [inSquare.y + inSquareHalf,inSquare.y - inSquareHalf]
		return self.containPoint(inSquareX[0],inSquareY[0]) and self.containPoint(inSquareX[1],inSquareY[1])

print 'Question 5 Test Cases\n'

s = Square(x=1,y=1,sideLength=2.0)
print s.getCenter()
print s.getSideLength()
print s.getArea()
print s.getPerimeter()
print s.containPoint(0,0)
print s.containPoint(0,-0.5)
print s.containPoint(1,1.5)
print s.containSquare(Square(x=1.5,y=1,sideLength=1))
print s.containSquare(Square(x=1.5,y=1,sideLength=1.1))

s2 = Square()
print s2.getCenter()
print s2.getSideLength()
print s2.getPerimeter()

print '\n'

##### Question 6

from libdw import sm

class Elevator(sm.SM):
	startState = 'First'

	def getNextValues(self,state,inp):
		if state == 'First':
			if inp == 'Up':
				return ('Second','Second')
			else:
				return ('First','First')
		elif state == 'Third':
			if inp == 'Down':
				return ('Second','Second')
			else:
				return ('Third','Third')
		else:
			if inp == 'Up':
				return ('Third','Third')
			else:
				return ('First','First')

print 'Question 6 Test Cases\n'

e = Elevator()
print e.transduce(['Up', 'Up', 'Up', 'Up', 'Down', 'Down', 'Down', 'Up'])

print '\n'

##### Question 7

def countNumOpenLocker(K):
	lockerState = [False]*K
	for i in xrange(1,K+1):
		for j in range(i-1,len(lockerState),i):
			lockerState[j] = not lockerState[j]
	
	openCount = 0
	for i in lockerState:
		if i:
			openCount += 1

	return openCount

print 'Question 7 Test Cases'

print countNumOpenLocker(6)
print countNumOpenLocker(2000)
print countNumOpenLocker(10)
print countNumOpenLocker(20)
print countNumOpenLocker(1000000)

##### One-liner!

import math

def countNumOpenLockerOne(K):
	return int(math.floor(math.sqrt(K)))

print 'Question 7 One Liner'

print countNumOpenLockerOne(6)
print countNumOpenLockerOne(2000)
print countNumOpenLockerOne(10)
print countNumOpenLockerOne(20)
print countNumOpenLockerOne(1000000)
