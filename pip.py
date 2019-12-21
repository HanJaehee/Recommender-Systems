import numpy as np
import pandas as pd
# df is your DataFrame
x = df['rating']
med = (x.max() + x.min())/2
def Agreement(r1, r2): # Agreement 구현
    if(r1 > med and r2 < med) or (r1<med and r2>med):
        return False
    else:
        return True
    
def Proximity(r1, r2): # Proximity 구현
    if Agreement(r1, r2):
        D = np.abs(r1-r2)
    else:
        D = np.abs(r1-r2)*2
    return pow((2*x.max()-x.min()+1)-D, 2)

def Impact(r1, r2): # Impact 구현
    tmp = (np.abs(r1-med)+1)*(np.abs(r2-med)+1)
    if Agreement(r1, r2):
        return tmp
    else:
        return 1/tmp
    
def Popularity(r1, r2, iid): # Popularity 구현
    tmp = df[df['iid'] == iid]
    mean = tmp['rating'].mean()
    if (r1>mean and r2>mean) or (r1<mean and r2<mean):
        return 1+pow((r1+r2)/2-mean, 2)
    else:
        return 1


def PIP(df): # PIP 본체 구현
    NumUsers = 25 # 총 유저수
    sim = np.full((NumUsers,NumUsers), 0.0)
    for u in range(0,NumUsers):
        for v in range(u+1, NumUsers):
           # print("u=",u, " v=",v)
            u1 = df[df['uid']==str(u+1)] # uid가 일치하는 행들을 추출
            u2 = df[df['uid']==str(v+1)]

            inter = pd.merge(u1['iid'], u2['iid'], on='iid') # u1과 u2가 일치하는 iid를 추출(교집합)
            similarity = 0
            for i in range(len(inter)):
                iid = inter[i:i+1]
                iid = pd.to_numeric(iid.iid)
                iid = iid[i] #index로 먹기때문에 인덱스를 i로 맞춰주면댐

                u1_rating = u1[u1['iid']==str(iid)]['rating'].iloc[0]
                u2_rating = u2[u2['iid']==str(iid)]['rating'].iloc[0]
                # user1의 레이팅 중 iid와 일치하는 행에서 rating값을 숫자로 리턴

                prox = Proximity(u1_rating, u2_rating)
                imp = Impact(u1_rating, u2_rating)
                pop = Popularity(u1_rating, u2_rating, iid)
                pip = prox*imp*pop
                similarity += pip # pip 연산
            sim[u,v] = similarity
            sim[v,u] = sim[u,v]
            #sim = sim/sim.max()
    return sim