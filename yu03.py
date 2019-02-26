# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:30:29 2018

@author: yujiaoliang
"""
import networkx as nx
import pylab 
import numpy as np
#自定义网络
#row=np.array([0,0,1,1,1,])
col=np.array([1,3,5,4,2,])

print('生成一个空的有向图')
G=nx.DiGraph()
print('为这个网络添加节点...')
for i in range(0,np.size(col)+1):
    G.add_node(i)
print('在网络中添加带权中的边...')
#for i in range(np.size(row)):
G.add_edges_from([(0,1)])
G.add_edges_from([(0,5)])
G.add_edges_from([(1,5)])
G.add_edges_from([(1,2)])
G.add_edges_from([(2,3)])
G.add_edges_from([(2,4)])
G.add_edges_from([(3,4)])
G.add_edges_from([(4,5)])
G.add_edges_from([(1,0)])
G.add_edges_from([(5,0)])
G.add_edges_from([(5,1)])
G.add_edges_from([(2,1)])
G.add_edges_from([(3,2)])
G.add_edges_from([(4,2)])
G.add_edges_from([(4,3)])
G.add_edges_from([(5,4)])




print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )
pylab.title('Self_Define Net',fontsize=15)
pylab.show()




'''
shortest_path function
'''

p=nx.all_shortest_paths(G,source=0,target=4)
for i in p:
#    for j in i:
       print('源节点为0，终点为3：', i)
# =============================================================================
# distance=nx.shortest_path_length(G,source=0,target=7)
# print('源节点为0，终点为7,最短距离：', distance)
# 
# p=nx.shortest_path(G,source=0) # target not specified
# print('只给定源节点0：', p[7])
# distance=nx.shortest_path_length(G,source=0) # target not specified
# print('只给定源节点0, 最短距离：', distance[7])
# 
# p=nx.shortest_path(G,target=7) # source not specified
# print('只给定终点7：', p[0])
# distance=nx.shortest_path_length(G,target=7)# source not specified
# print('只给定终点7，最短距离：', distance[0])
# 
# p=nx.shortest_path(G) # source,target not specified
# print('源节点，终点都为给定：', p[0][7])
# =============================================================================
# =============================================================================
# distance=nx.shortest_path_length(G) # source,target not specified
# 
# print('源节点，终点都为给定，最短距离：', distance[0][7])
# =============================================================================
