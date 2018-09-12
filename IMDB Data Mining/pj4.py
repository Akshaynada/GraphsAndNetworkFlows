#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:03:50 2018

@author: zonghengma
"""

import igraph
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

#1uestion 1

lines = []
with open("/Users/zonghengma/Documents/UCLA/ee232/project4/actor_movies.txt", "rb") as f:
    for line in f:
        lines.append(line.decode('unicode_escape'))
        
with open("/Users/zonghengma/Documents/UCLA/ee232/project4/actress_movies.txt", "rb") as f:
    for line in f:
        lines.append(line.decode('unicode_escape'))
        
dataset = dict()
movie_to_actor = dict()

for i in range(len(lines)):
    split = lines[i].find("\t\t")
    name = lines[i][:split]
    movies = lines[i][split+2:]
    movies = movies.strip().split("\t\t")
    cleaned_movies = set()
    for m in movies:
        cleaned_movies.add(re.sub("\([^0-9V]+\)", "", m.strip()).strip())
    if len(cleaned_movies) >= 10:
        dataset[name] = cleaned_movies
    
for i in dataset:
    for m in dataset[i]:
        if m not in movie_to_actor:
            movie_to_actor[m] = set()
        movie_to_actor[m].add(i)
        
print(len(dataset))
print(len(movie_to_actor))

#question 2
edge_list = dict()
for i in tqdm(dataset):
    for m in dataset[i]:
        for j in movie_to_actor[m]:
            if i not in edge_list:
                edge_list[i] = dict()
            if j not in edge_list[i]:
                edge_list[i][j] = 1
            else:
                edge_list[i][j] += 1

for i in edge_list:
    for j in edge_list[i]:
        edge_list[i][j] = edge_list[i][j] / len(dataset[i])
                
with open("/Users/zonghengma/Documents/UCLA/ee232/project4/edgelist.txt", "w") as f:
    for i in tqdm(edge_list):
        for j in edge_list[i]:
            f.write((i.replace(" ", "_") + "\t" + j.replace(" ", "_") + "\t" + str(edge_list[i][j]) + "\n"))
            
with open("/Users/zonghengma/Documents/UCLA/ee232/project4/edgelist_short.txt", "wb") as f:
    for i in tqdm(edge_list):
        for j in edge_list[i]:
            f.write((i + "\t" + j + "\t" + str(edge_list[i][j]) + "\n").encode())
        break

#with open("/Users/zonghengma/Documents/UCLA/ee232/project4/edgelist.txt", "r") as f:
g = igraph.Graph.Read_Ncol("/Users/zonghengma/Documents/UCLA/ee232/project4/edgelist.txt", directed=True, names=True, weights=True)

plt.hist(g.degree(mode='in'), bins=50)
