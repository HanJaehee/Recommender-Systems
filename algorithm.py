# -*- coding:utf-8 -*-
import numpy as np

def COS_item(a): #item-based COS
    a = a.T #편의상 축을 바꿈
    Numitems = np.size(a, axis=0)
    Sim = np.full((Numitems, Numitems), 0.0)
    for i in range(0, Numitems):
        arridx_i = np.where(a[i,] == 99)
        for j in range(i+1, Numitems):
            arridx_j = np.where(a[j,] == 99)
            arridx = np.unique(np.concatenate((arridx_i, arridx_j), axis = None))

            U = np.delete(a[i, ], arridx) # 위의 NULL 인덱스들을 원래의 matrix에서 제거
            V = np.delete(a[j, ], arridx)

            InnerDot = np.dot(U, V) # 위의 각각 계산한 U,V 배열을 내적계산
            NormU = np.linalg.norm(U) # 각 요소 제곱의 합에 루트한 값
            NormV = np.linalg.norm(V)

            if NormU == 0 or NormV==0:
                Sim[i,j] = 0
            else:
                Sim[i,j] = InnerDot/(NormU * NormV) # 계산
            Sim[j,i] = Sim[i,j] # user 1,user2 와 user2,user1과의 similarity는 같으므로 값 복사
    
    return Sim.T #다시 축을 바꿔줌

def basic_CF(mat, sim, k):
    predicted_rating = np.array([[0.0 for col in range(10)] for row in range(5)])

    if(sim == 'COS'):
        sim = COS(mat)
    if sim == 'PCC':
        sim = PCC(mat)
    
    k_neighbors = np.argsort(-sim)
    k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)

    NumUsers = np.size(mat, axis=0)

    for u in range(0, NumUsers):
        list_sim = sim[u, k_neighbors[u,]]
        list_rating = mat[k_neighbors[u,],].astype('float64')
        predicted_rating = np.sum(list_sim.reshape(-1,1) * list_rating, axis=0)/ np.sum(list_sim)
    
    return predicted_rating

def CF_baseline(mat, sim, k): # (User-based) 평점을 예측하고자 하는 사용자와 유사도가 큰 k명의 사용자를 대상으로 예측 레이팅을 구한다.
    predicted_rating = np.array([[0.0 for col in range(100)] for row in range(1000)])

    item_mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=0) #item 평균
    user_mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=1) #user 평균
    all_mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=None) # 전체평균

    if sim =='COS':
        Sim = COS(mat)
    elif sim == 'PCC':
        Sim = PCC(mat)

    k_neighbors = np.argsort(-Sim) #큰 수부터 인덱싱한 배열
    k_neighbors = np.delete(k_neighbors, np.s_[k:], axis=1) #k_neighbors 배열 중 각 배열의 k 인덱스 이후의 요소들을 모두 삭제

    NumUsers = np.size(mat, axis=0) #유저 수 반환

    for u in range(0, NumUsers):
        v = k_neighbors[u,] # v들을 의미하는 k_neighbors[u,]를 보기 편하도록 변수로 바꾸어줌
        list_sim = Sim[u, v] # sim(u,v)를 의미
        list_rating = mat[v, ].astype('float64') # r(v,i)를 의미하며 아이템이 여러개이기 때문에 2차원행렬로 값이 나타남
        list_rating[list_rating==99] = np.nan # Null이 제거되지않은 행렬이기 때문에 nan처리
        
        base_u = user_mean[u,] + item_mean - all_mean #식을 풀면 user평균 + item 평균 - 전체 rating 평균
        base_v = user_mean[v,].reshape(-1,1) + item_mean - all_mean
        denominator = np.sum(list_sim)
        numerator = np.sum(list_sim.reshape(-1,1) *(list_rating - base_v), axis=0)
        predicted_rating[u, ] = base_u + numerator / denominator

    return predicted_rating

def CF_baseline_i(mat, k):# CF with baseline (Item-based)
    predicted_rating = np.array([[0.0 for col in range(1000)] for row in range(100)]) #축을 바꿔 계산할 것이기 때문에 행과 열의 크기를 바꿔 생성

    item_mean = np.nanmean(np.where(mat != 99 , mat, np.nan), axis=0)
    user_mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=1)
    all_mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=None) #기존과 동일 

    Sim = COS_item(mat) # COS만을 Item-based로 구현해 COS로 기본사용..합니다

    k_neighbors = np.argsort(-Sim)
    k_neighbors = np.delete(k_neighbors, np.s_[k:], axis=1) # 기존과 동일
   
    Numitems = np.size(mat, axis=1) # Item의 갯수

    for i in range(0, Numitems):
        j = k_neighbors[i,]
        list_sim = Sim[i,j]
        mat_T = mat.T # 아이템을 선택한 user들을 불러오기 위해 기존 행렬의 축을 바꿔줍니다.
        list_rating = mat_T[j,].astype('float64') #r(u,j)를 의미합니다
        list_rating[list_rating==99] = np.nan #Null값인 99를 nan처리


        base_i = item_mean[i,] + user_mean - all_mean
        base_j = item_mean[j,].reshape(-1,1) + user_mean - all_mean
        denominator = np.sum(list_sim)
        numerator = np.sum(list_sim.reshape(-1,1)*(list_rating - base_j), axis=0)
        predicted_rating[i,] = base_i + numerator /denominator #기존 계산들과 동일

    return predicted_rating.T #행과열 다시 바꿔서 원래 모양대로 return

def basic_mean(mat, sim, k):
    predicted_rating = np.array([[0.0 for col in range(100)] for row in range(1000)])

    mean = np.nanmean(np.where(mat != 99, mat, np.nan), axis=1)

    if sim == 'COS':
        Sim = COS(mat)
    elif sim == 'PCC':
        Sim = PCC(mat)
    
    k_neighbors = np.argsort(-Sim) #큰 수부터 인덱싱한 배열
    k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) #k_neighbors 배열 중 각 배열의 k 인덱스 이후의 요소들을 모두 삭제

    NumUsers = np.size(mat, axis=0) #유저 수 반환

    for u in range(0, NumUsers):
        v = k_neighbors[u,]
        list_sim = Sim[u, v]
        list_rating = mat[v, ].astype('float64')
        list_rating[list_rating==99] = np.nan # nan처리되지 않은 행렬이기 때문에 nan처리
        list_mean = mean[k_neighbors[u,],]

        denominator = np.sum(list_sim)
        numerator = np.sum(list_sim.reshape(-1,1) *(list_rating - list_mean.reshape(-1,1)), axis=0)
        predicted_rating[u, ] = mean[u] + numerator / denominator

    return predicted_rating
    
def basic_zscore(mat, sim, k):
    predicted_rating = np.array([[0.0 for col in range(10)] for row in range(5)])

    mean = np.nanmean(np.where(mat!=0, mat, np.nan), axis=1)
    std = np.nanstd(np.where(mat!=0, mat, np.nan), axis=1)

    if(sim == 'COS'):
        sim = COS(mat)
    if sim == 'PCC':
        sim = PCC(mat)
    
    k_neighbors = np.argsort(-sim)
    k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)

    NumUsers = np.size(mat, axis=0)

    for u in range(0, NumUsers):
        list_sim = sim[u, k_neighbors[u,]]
        list_rating = mat[k_neighbors[u,],].astype('float64')
        list_mean = mean[k_neighbors[u,],]
        list_std = std[k_neighbors[u,],]

        denominator = np.sum(list_sim)
        numerator = np.sum((list_sim.reshape(-1,1)*(list_rating-list_mean.reshape(-1,1))/list_std.reshape(-1,1)), axis=0)
        predicted_rating[u,] = mean[u] + std[u]*numerator/denominator

    return predicted_rating


def CPCC(a): # Constrained PCC (COS와 PCC를 조합, ))
    NumUsers = np.size(a, axis=0) # x축의 size, 여기서는 row의 갯수를 의미한다.
    Sim = np.full((NumUsers, NumUsers), 0.0)
    
    median = np.nanmedian(np.where(a!=99, a, np.nan), axis=1) # 각 사용자 ratings의 중앙값 (오름차순으로 정렬 후 중간에 있는 값을 의미)

    for u in range(0, NumUsers):
        for v in range(u, NumUsers):
            arridx_u = np.where(a[u, ] == 99)
            arridx_v = np.where(a[v, ] == 99)
            arridx = np.concatenate((arridx_u, arridx_v), axis = None)

            U = np.delete(a[u, ], arridx)
            V = np.delete(a[v, ], arridx)
            U = U - median[u]
            V = V - median[v]

            InnerDot = np.dot(U, V)
            NormU = np.linalg.norm(U)
            NormY = np.linalg.norm(V)
            Sim[u,v] = InnerDot/(NormU * NormY)
            Sim[v,u] = Sim[u,v]
            Sim[np.isnan(Sim)] = -1
    return Sim

    
def PCC(a): # Pearson Correlation Coefficient (COS와 비슷한 부분이 많아 같은 부분은 설명 생략)
    NumUsers = np.size(a, axis=0)
    Sim = np.full((NumUsers, NumUsers), 0.0)

    #np.where(a!=99, a, np.nan) => a!=99에 대한 True는 그대로 a값을 넣어주고, False는 nan을 넣어준다.
    mean = np.nanmean(np.where(a!=99, a, np.nan), axis=1) # 각 사용자 ratings 의 평균값 (산술평균을 의미)
    
    for u in range(0, NumUsers):
        arridx_u = np.where(a[u, ] == 99)
        for v in range(u, NumUsers):
            arridx_v = np.where(a[v, ] == 99)
            arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis=None))

            U = np.delete(a[u,], arridx) - mean[u]  # NULL이 아닌 요소 각각에 평균값을 빼줌
            V = np.delete(a[v, ], arridx) - mean[v]

            InnerDot = np.dot(U,V) # 평균값을 뺀값에 대한 내적
            NormU = np.linalg.norm(U) # 평균값을 뺀값에 대한 각 요소 제곱의 합에 루트한 값
            NormV = np.linalg.norm(V)

            if NormU == 0 or NormV == 0:
                Sim[u,v] = 0
            else:
                Sim[u,v] = InnerDot/(NormU*NormV)
            Sim[v,u] = Sim[u,v]
    return Sim


def COS(a): # Cosine Similarity (w/o Null, 두 사용자의 rating 벡터 사이의 각을 계산해 그 각을 similarity의 정도로 본다.
    NumUsers = np.size(a, axis=0) # row의 갯수
    Sim = np.full((NumUsers, NumUsers), 0.0) # user대 user의 similarity를 비교하므로 user x user 크기 배열 생성
    for u in range(0, NumUsers):
        arridx_u = np.where(a[u, ] == 99) # data중 NULL을 의미하는 값의 인덱스 찾음(jester Null = 99)
        for v in range(u+1, NumUsers): # u부터 시작함으로써 같은 계산을 두번씩하지 않음
            arridx_v = np.where(a[v, ] == 99)
            arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis = None)) # axis=None으로 설정할시 하나의 배열로 합쳐짐
            #np.unique로 중복 없애줌.
            U = np.delete(a[u, ], arridx) # 위의 NULL 인덱스들을 원래의 matrix에서 제거
            V = np.delete(a[v, ], arridx)

            InnerDot = np.dot(U, V) # 위의 각각 계산한 U,V 배열을 내적계산
            NormU = np.linalg.norm(U) # 각 요소 제곱의 합에 루트한 값
            NormV = np.linalg.norm(V)

            if NormU == 0 or NormV==0:
                Sim[u,v] = 0
            else:
                Sim[u,v] = InnerDot/(NormU * NormV) # 계산
            Sim[v,u] = Sim[u,v] # user 1,user2 와 user2,user1과의 similarity는 같으므로 값 복사
    return Sim
