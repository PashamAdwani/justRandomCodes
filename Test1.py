
#Lib
import numpy as np
import pandas as pd
import matplotlib
import random
import time


#Variables

nb=int(input('Number of Black Queens:'))
nw=int(input('Number of White Queens:'))

#For the x,y position across the chessboard
BQ=np.zeros((nb,2))
WQ=np.zeros((nw,2))
#whenever there is a low score
bestBQ=np.zeros((nb,2))
bestWQ=np.zeros((nw,2))

N=int(input('Insert N dimensions of the chessboard:'))
M=int(input('Insert M dimensions of the chessboard:'))

#Score is the number of Queens safe
scorew=0
scoreb=0

#1 means white queen, 100 means black queen 
chessboard=np.zeros((N,M))

#Calculate Score
def CalculateRiskScore(posx,posy,RivalQueen,rn,score): #rn is number of rivalQueens
    for i in range(rn):
        if(RivalQueen[i][0]-posx == RivalQueen[i][1]-posy):
            score=score+1
        elif(RivalQueen[i][1]==posy):
            score=score+1
        elif(RivalQueen[i][0]==posx ):
            score=score+1
    return score                    
                    

def randomAssign(n,m,Queen,nq):
    for i in range(nq):
        x=random.randint(0,n)
        y=random.randint(0,m)
        Queen[i][0]=x
        Queen[i][1]=y
    return Queen


#Main

BQ=randomAssign(N,M,BQ,nb)
WQ=randomAssign(N,M,WQ,nw)
scoreb=0
for i in range(nb):
    scoreb=CalculateRiskScore(BQ[i][0],BQ[i][1],WQ,nw,scoreb)

scorew=0
for i in range(nw):
    scorew=CalculateRiskScore(WQ[i][0],WQ[i][1],BQ,nb,scorew)

score=scorew+scoreb




timeStart=time.time()
timeElapsed=0
iterations=0
tmax=60
leastScore=score
solutionFound=0 #solutionFound=1 if score=0 that is the 
while(timeElapsed<=tmax and solutionFound==0):
    iterations=iterations+1
    BQ=randomAssign(N,M,BQ,nb)
    WQ=randomAssign(N,M,WQ,nw)
    scoreb=0
    for i in range(nb):
        scoreb=CalculateRiskScore(BQ[i][0],BQ[i][1],WQ,nw,scoreb)
    scorew=0
    for i in range(nw):
        scorew=CalculateRiskScore(WQ[i][0],WQ[i][1],BQ,nb,scorew)
    score=scoreb+scorew
    if(score==0):
        solutionFound=1
        print('Solution Found')
    if(score<leastScore):
        bestBQ=BQ
        bestWQ=WQ
    timeElapsed=time.time()-timeStart    

        
print('score',score,' iteration', iterations, 'time', timeElapsed)
print('BQ',bestBQ)
print('White',bestWQ)

dispCB=np.zeros((N,M))
for i in range(N):
    for j in range(M):
        for k in range(nw):
            if(i==bestWQ[k][0] and j==bestWQ[k][1]):
                dispCB[i][j]=10
        for k in range(nb):
            if(i==bestBQ[k][0] and j==bestBQ[k][1]):
                dispCB[i][j]=-10

print(dispCB)                
