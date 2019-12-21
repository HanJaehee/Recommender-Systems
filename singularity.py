import numpy as np
import pandas as pd
# df is your DataFrame

def get_PN(iid): # S_p와 S_n 구함
    x = df[df['iid']==str(iid)]
    #print(x.shape)
    if not x.shape[0]: #일치하는 iid가 없을 경우 0 return
        return 0
    P = []
    N = []
    U=25 # 총 유저수
    for i in range(0, x.shape[0]):
        tmp = x[i:i+1]
        if tmp['rating'].iloc[0] > 0.5 :
            P.append(tmp['uid'].iloc[0]) # P에 해당하는 인원 P에 추가
        else:
            N.append(tmp['uid'].iloc[0]) # N에 해당하는 인원 N에 추가
    S_p = 1-(len(P)/U) # P의 길이 = P 유저 수 
    S_n = 1-(len(N)/U) # N의 길이 = N 유저 수
    tmp = [[iid, S_p, S_n]]
    return tmp # iid와 iid의 S_p, S_n값을 담은 리스트 리턴

S = pd.DataFrame(columns=["iid", "S_p", "S_n"]) #  get_PN 이용, iid별 S_p와 S_n를 구한다.
for i in range(0,150):
    result = get_PN(i)
    if result == 0: #반환값이 0 이면 데이터 프레임에 담지 않음
        continue
    tmp = pd.DataFrame(data=result, columns = ["iid", "S_p", "S_n"])
    S = S.append(tmp)
    
def Singularity(df):
    NumUsers = 25 #총 유저수
    sim = np.full((NumUsers,NumUsers), 0.0)
    for u in range(0,NumUsers):
        for v in range(u+1, NumUsers):
            u1 = df[df['uid']==str(u+1)] # uid가 일치하는 행들을 추출
            u2 = df[df['uid']==str(v+1)]

            inter = pd.merge(u1['iid'], u2['iid'], on='iid') # u1과 u2가 일치하는 iid를 추출(교집합)
            A=0
            B=0
            C=0
            A_val=0
            B_val=0
            C_val=0
            for i in range(len(inter)):
                iid = inter[i:i+1] #iid 교집합중 하나씩 추출
                iid = pd.to_numeric(iid.iid)
                iid = iid[i] #index로 먹기때문에 인덱스를 i로 맞춰주면댐
                S_p = S[S['iid']==iid]['S_p'].iloc[0]
                S_n = S[S['iid']==iid]['S_n'].iloc[0]
                #print("u= ",u,"v= ",v,"i= ",i)
                u1_rating = float(u1[u1['iid']==str(iid)]['rating'].iloc[0]) #각 레이팅 추출
                u2_rating = float(u2[u2['iid']==str(iid)]['rating'].iloc[0])
                tmp = u1_rating - u2_rating
                if u1_rating > 0 and u2_rating > 0:#공식에 맞게 설정
                    A += 1
                    A_val += (1-tmp*tmp)*S_p*S_p
                elif u1_rating <=0 and u2_rating <= 0:
                    B += 1
                    B_val += (1-tmp*tmp)*S_n*S_n
                elif (u1_rating > 0 and u2_rating <=0) or (u1_rating <= 0 and u2_rating >0):
                    C +=1
                    C_val += (1-tmp*tmp)*S_p*S_n

            if A==0 and B==0 and C==0: #각 A,B,C의 경우들 나열해 결과값 조정
                result = '0'
            elif A==0 and B==0:
                result = C_val/C
            elif B==0 and C==0:
                result = A_val/A
            elif A==0 and C==0:
                result = B_val/B
            elif A==0:
                result = (B_val/B + C_val/C)/2
            elif B==0:
                result = (A_val/A + C_val/C)/2
            elif C==0:
                result = (A_val/A + B_val/B)/2
            else:
                result = (A_val/A + B_val/B + C_val/C)/3
            sim[u,v] = result
            sim[v,u] = sim[u,v]
    return sim