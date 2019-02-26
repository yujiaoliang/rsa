# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:58:01 2018

@author: yujiaoliang
"""
import scipy.io as sio
import numpy as np
import yu03_1 as y301
import yu01 as y01
import yu02 as y02
import xlwt
# =============================================================================
# def str2int(string):
#     a = list(map(int, str(string))) 
#     return a
# 
# 
# if __name__ == '__main__':
#     string = "12244"
#     print(str2int(string))
# 
# =============================================================================

#    print(dic)
    
def renlu_1(path_ar):
    index_d ={}     #索引列表
    index_l =[]
    key = 0
    dic = {'01':"0",'13':"1",'31':'1',"35":"2",'53':'2','54':"3",'45':'3','42':"4","24":'4','02':"5",'20':'5','12':"6",'21':'6','34':"7",'43':'7'}
    for string in  path_ar:
        for i in range(len(string)-1):
            indice = string[i:i+2]
            index = int(dic[indice])
            index_l.append(index)
#        print(index_l)
        index_d[key] = index_l
        key+=1
        index_l = [] 
    return(index_d)

    
def renlu_2(index_d):       
    temp_a = []
    temp_b = []
    temp_c = []
    conti = []
    continuous = []
    conindex = []
    cin_list = []
    cin_index_list = []
    con_index_list1 = []
    con_index_list2 = []
    Final = [0,0,0,0,0,0]
    j = 0
    data = sio.loadmat('C:\\Users\\yujiaoliang\\Desktop\\matlab.mat')
    datain = data['dataInMatrix']
    datain = np.array(datain).reshape(3000,8,6)
#    print(datain[0][0])
    for i in range(3000):   #合并路径
        temp_a = datain[i]
        Final = [0,0,0,0,0,0]
        print(temp_a)
        temp_b = []
        for item in index_d.values():
            Final = [0,0,0,0,0,0]
            while j<=7:
                if j in item:
                   Final = list(map(lambda x,y:x or y, Final,temp_a[j]))
                j+=1
            j = 0
            print(Final) 
            temp_b.append(Final)
        print(temp_b)
        temp_c.append(temp_b)
#    print (temp_c)
    for m in range(3000):
        conti = []
        temp_d = temp_c[m]
        for n in range(4):
            con = y01.lian0(temp_d[n])
            conti.append(con)  
        for r in range(4):
            continuous.append(conti[r][1])
            conindex.append(conti[r][0])
#    print(continuous)
#    print(conindex)
    continu = [continuous[i:i+4] for i in range(0,len(continuous),4)]
    conindex_l = [conindex[i:i+4] for i in range(0,len(conindex),4)]
    for s in range(3000):
        continuou = continu[s]
#        print(continuou)
        cin = min(continuou)
        cin_index = continuou.index(min(continuou))
        con_index = conindex_l[s][cin_index]
        con_index.tolist()
        con_index1 = int(con_index[0])
        con_index2 = int(con_index[1])
#        con_index.tolist()
#        print(con_index1,type(con_index1))
        cin_list.append(cin)
        cin_index_list.append(cin_index)
        con_index_list1.append(con_index1)
        con_index_list2.append(con_index2)
    print(cin_list,cin_index_list,con_index_list1,con_index_list2)
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet("频谱",cell_overwrite_ok=True)
    for j in range(3000):    
           worksheet.write(j,0,cin_list[j])  
           worksheet.write(j,1,cin_index_list[j]) 
           worksheet.write(j,2,con_index_list1[j])
           worksheet.write(j,3,con_index_list2[j]) 
    workbook.save("C:\\Users\\yujiaoliang\\Desktop\\result.xls")
#    y02.writeform(cin_list,7,0)
#    y02.writeform(cin_index_list,7,1)
#    y02.writeform(con_index_list2,7,3)
     
#    y02.writeform(con_index_list,7)
#        y02.writeform(t,1,cin_index_list[t])
#        y02.writeform(t,3,con_index_list)
#        

        
        
            
if __name__ =='__main__':
#    xuanlu()
    h = y301.Graph()
    path_ar = h.zhu()
    print(path_ar)
    index_d = renlu_1(path_ar)
    print(index_d)
    renlu_2(index_d)
    