# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:32:26 2019

@author: yujiaoliang
"""

import csv
#import sys
#sys.path.append('D:\\teris')
#import yu01 as y1
import itertools
import numpy as np
import xlwt

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
    lyst = list(map(int,lyst))
    ar = []
    for i  in range(n-3):
        if sum(lyst[i:i+4])==0:
            ar.append([i,i+1,i+2,i+3])
    a =np.array(ar)
    a=a.reshape(-1,4)
    lxd_ar=[]
    row = len(a)-1
    j = 0
    lystc = lyst
    while j <= row:
        h_1,h_2,h_3,h_4 = a[j][0],a[j][1],a[j][2],a[j][3]
        lystc[h_1],lystc[h_2],lystc[h_3],lystc[h_4] = 1,1,1,1
        #print(lystc)
        lxd = lianxudu(lystc)
        lxd_ar.append(lxd)
        lystc[h_1],lystc[h_2],lystc[h_3],lystc[h_4] = 0,0,0,0
        j+=1
    if len(lxd_ar)== 0:
       p = 999
       q = 9
    else:
       q = min(lxd_ar)
       p_1 = lxd_ar.index(min(lxd_ar))
       p = a[p_1][0]
    return p,q


if __name__ == '__main__':
    x = []
    with open (r"C:\\Users\\yujiaoliang\\Desktop\\three.csv") as csvfile1:
        csv_reader = csv.reader(csvfile1)
        for row in csv_reader:
            x.append(row)
    l = len(x)
    print(lian0(x[0])[0])
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet("频谱",cell_overwrite_ok=True)
    for i in range(l):
        worksheet.write(i,0,int(lian0(x[i])[0])) 
      
    workbook.save("C:\\Users\\yujiaoliang\\Desktop\\result2.xls")
        
    
    
    
    