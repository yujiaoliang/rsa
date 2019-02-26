# import sys
# sys.path.append('D:\\python\\lib')
'''
计算连续度

'''
import scipy.io as sio
import numpy as np
import itertools
def lianxudu(x):
    if 0 in x:
        max_group = max([len(list(j)) for i, j in itertools.groupby(x) if i == 0])
        total = x.count(0)
        lxd =1-max_group/total
    else:
        lxd = 0 
    return lxd 

def lian0(lyst):
    n=len(lyst)-1
    ar = []
    for i  in range(n):
        if lyst[i] + lyst[i+1]==0:
            ar.append(i)
            ar.append(i+1)
    a=np.array(ar)
    a=a.reshape(-1,2)
    print(a)
    lxd_ar=[]
    row = len(a)-1
    j = 0
    lystc = lyst
    while j <= row:
        h,l = a[j][0],a[j][1]
        lystc[h],lystc[l] = 1,1
        #print(lystc)
        lxd = lianxudu(lystc)
        lxd_ar.append(lxd)
        lystc[h], lystc[l] = 0,0
        j+=1
#    print(lxd_ar)
    if len(lxd_ar)== 0:
       p = np.array([9, 9])
       q = 9
    else:
       q = min(lxd_ar)
    #lian0_array =[]
    #lian0_array.append(q)
    #lian0_array.append(lxd_ar.index(min(lxd_ar)))
       p_1 = lxd_ar.index(min(lxd_ar))
       p = a[p_1]
    return p,q

if __name__=='__main__':
    x = [0,1,0,0,0,0]
            
    print(lian0(x))
# =============================================================================
#     data_in = datain
#     result = []
#     for i in range(2000):
#         lyst = data_in[i]
#         lian0(lyst)
#         result.append(q)
#         result.append(p)
#     result_1 = np.array(result).reshape(-1,2)
#     print(result_1)
# =============================================================================
#    lyst=[0,0,1,0,0,0,0,1,1,0]
#    print(lian0(lyst))
    