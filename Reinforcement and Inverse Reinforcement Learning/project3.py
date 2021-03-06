# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:20:49 2018

@author: cyrus
"""

#import libraries needed

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
reward_1 = np.zeros(shape=(10,10))
reward_2 = np.zeros(shape=(10,10))

reward_1[9,9] = 1
reward_2[1,4] = -100
reward_2[1,5] = -100
reward_2[1,6] = -100
reward_2[2,4] = -100
reward_2[2,6] = -100
reward_2[3,4] = -100
reward_2[3,6] = -100
reward_2[3,7] = -100
reward_2[3,8] = -100
reward_2[4,4] = -100
reward_2[4,8] = -100
reward_2[5,4] = -100
reward_2[5,8] = -100
reward_2[6,4] = -100
reward_2[6,8] = -100
reward_2[7,6] = -100
reward_2[7,7] = -100
reward_2[7,8] = -100
reward_2[8,6] = -100
reward_2[9,9] = 10

reward_1_heat_map = plt.imshow(reward_1,cmap=plt.cm.Reds)

reward_2_heat_map = plt.imshow(reward_2,cmap=plt.cm.Reds)
plt.show()

transition_probabilities_matrix = np.zeros(shape=(4,100,100))

non_boundary_states = list(range(11,19)).extend(list(range(21,28)))
#non-boundary cases

actions = ('right', 'left' , 'up', 'down')
w = 0.1

#transition probabilities for non-boundary states
for action in actions:
    action_index = actions.index(action)
    #non-boundary states
    for i in range(11,82,10):
        for j in range(8):
            current_state = i + j
            if action_index == 0:
                transition_probabilities_matrix[action_index,current_state,current_state - 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state+ 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state - 10] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 10] = 1 - w + w / 4
            elif action_index == 1:
                transition_probabilities_matrix[action_index,current_state,current_state - 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state - 10] = 1 - w + w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 10] = w / 4
            elif action_index == 2:
                transition_probabilities_matrix[action_index,current_state,current_state - 1] = 1 - w + w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state - 10] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 10] = w / 4
            else:
                transition_probabilities_matrix[action_index,current_state,current_state - 1] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 1] = 1 - w + w / 4
                transition_probabilities_matrix[action_index,current_state,current_state - 10] = w / 4
                transition_probabilities_matrix[action_index,current_state,current_state + 10] = w / 4

    #corner cases
    
    #top left corner
    if action_index == 0:
        transition_probabilities_matrix[action_index,0,0] = w/4 + w/4
        transition_probabilities_matrix[action_index,0,1] = w/4
        transition_probabilities_matrix[action_index,0,10] = 1 - w + w/4
    elif action_index == 1:
        transition_probabilities_matrix[action_index,0,0] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,0,1] = w / 4
        transition_probabilities_matrix[action_index,0,10] = w / 4
    elif action_index == 2:
        transition_probabilities_matrix[action_index,0,0] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,0,1] = w/4
        transition_probabilities_matrix[action_index,0,10] = w/4
    elif action_index == 3:
        transition_probabilities_matrix[action_index,0,0] = w/4 + w/4
        transition_probabilities_matrix[action_index,0,1] = 1 - w + w/4
        transition_probabilities_matrix[action_index,0,10] = w/4
    
    
    #top right corner
    if action_index == 0:
        transition_probabilities_matrix[action_index,90,90] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,90,91] = w/4
        transition_probabilities_matrix[action_index,90,80] = w/4
    elif action_index == 1:
        transition_probabilities_matrix[action_index,90,90] = w/4 + w/4
        transition_probabilities_matrix[action_index,90,91] = w / 4
        transition_probabilities_matrix[action_index,90,80] = 1 - w + w/4
    elif action_index == 2:
        transition_probabilities_matrix[action_index,90,90] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,90,91] = w/4
        transition_probabilities_matrix[action_index,90,80] = w/4
    elif action_index == 3:
        transition_probabilities_matrix[action_index,90,90] = w/4 + w/4
        transition_probabilities_matrix[action_index,90,91] = 1 - w + w/4
        transition_probabilities_matrix[action_index,90,80] = w/4
        
    #bottom left corner
    if action_index == 0:
        transition_probabilities_matrix[action_index,9,9] = w/4 + w/4
        transition_probabilities_matrix[action_index,9,8] = w/4
        transition_probabilities_matrix[action_index,9,19] = 1-w + w/4
    elif action_index == 1:
        transition_probabilities_matrix[action_index,9,9] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,9,8] = w / 4
        transition_probabilities_matrix[action_index,9,19] = w / 4
    elif action_index == 2:
        transition_probabilities_matrix[action_index,9,9] = w/4 + w/4
        transition_probabilities_matrix[action_index,9,8] = 1 - w + w/4
        transition_probabilities_matrix[action_index,9,19] = w/4
    elif action_index == 3:
        transition_probabilities_matrix[action_index,9,9] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,9,8] = w/4
        transition_probabilities_matrix[action_index,9,19] = w/4
    
    #bottom right corner
    if action_index == 0:
        transition_probabilities_matrix[action_index,99,99] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,99,98] = w/4
        transition_probabilities_matrix[action_index,99,89] = w/4
    elif action_index == 1:
        transition_probabilities_matrix[action_index,99,99] = w/4 + w/4
        transition_probabilities_matrix[action_index,99,98] = w / 4
        transition_probabilities_matrix[action_index,99,89] = 1 - w + w/4
    elif action_index == 2:
        transition_probabilities_matrix[action_index,99,99] = w/4 + w/4
        transition_probabilities_matrix[action_index,99,98] = 1 - w + w/4
        transition_probabilities_matrix[action_index,99,89] = w/4
    elif action_index == 3:
        transition_probabilities_matrix[action_index,99,99] = 1 - w + w/4 + w/4
        transition_probabilities_matrix[action_index,99,98] = w/4
        transition_probabilities_matrix[action_index,99,89] = w/4

    
    #top edge
    for i in range(10,81,10):
        if action_index == 0:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 1:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 2:
            transition_probabilities_matrix[action_index,i,i] = 1- w + w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        else:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = 1 - w + w/4
    
    #bottom edge
    for i in range(19,90,10):
        if action_index == 0:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
        elif action_index == 1:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4            
        elif action_index == 2:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = 1 - w + w/4
        else:
            transition_probabilities_matrix[action_index,i,i] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
    
    #left edge
    for i in range(1,9):
        if action_index == 0:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 1:
            transition_probabilities_matrix[action_index,i,i] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 2:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = 1- w + w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 3:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
            transition_probabilities_matrix[action_index,i,i + 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = 1 - w + w/4
            
    
    #right edge
    for i in range(91,99):
        if action_index == 0:
            transition_probabilities_matrix[action_index,i,i] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 1:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        elif action_index == 2:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = 1 - w + w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = w/4
        else:
            transition_probabilities_matrix[action_index,i,i] = w/4
            transition_probabilities_matrix[action_index,i,i - 1] = w/4
            transition_probabilities_matrix[action_index,i,i - 10] = w/4
            transition_probabilities_matrix[action_index,i,i + 1] = 1 - w + w/4
            
            
    
#optimal state value function
epsilon = 0.01
V = [0] * 100
delta = math.inf
discount_factor = 0.8

while delta > epsilon:
    delta = 0
    for s in range(100):
        v = V[s]
        temp = []
        for action in range(4):
            sums = 0
            for s_prime in range(100):
                sums += transition_probabilities_matrix[action][s][s_prime] * (reward_1.flatten(order='F')[s_prime] + discount_factor * V[s_prime])
            
            temp.append(sums)
        
        V[s] = max(temp)
        delta = max([delta, abs(v - V[s])])
        

best_action = []
for s in range(100):
    temp = []
    for action in range(4):
        sums = 0
        for s_prime in range(100):
            sums += transition_probabilities_matrix[action][s][s_prime] * (reward_1.flatten(order='F')[s_prime] + discount_factor * V[s_prime])
            
        temp.append(sums)
    
    best_action.append(np.argmax(temp))
    


V_matrix = np.array(V).reshape((10,10))

fig,ax = plt.subplots()
for i in range(10):
    for j in range(10):
        c = V_matrix[i][j]
        ax.text(i + 0.5, j + 0.5, str(round(c,3)), va='center',ha='center')

plt.matshow(V_matrix,cmap='jet')
plt.colorbar()
ax.set_xlim(min_val,max_val)
ax.set_ylim(max_val,min_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(np.arange(max_val))
ax.grid()

best_action_matrix = np.array(best_action).reshape((10,10))
fig,ax = plt.subplots()
for i in range(10):
    for j in range(10):
        c = best_action_matrix[i][j]
        action = actions[c]
        print(action)
        
        if action == 'up':
            ax.text(i + 0.5, j + 0.5, u'\u2191', va='center', ha='center')
        elif action == 'down':
            ax.text(i + 0.5, j + 0.5, u'\u2193', va='center', ha='center')
        elif action == 'right':
            ax.text(i + 0.5, j + 0.5, u"\u2192", va='center', ha='center')
        else:
            ax.text(i + 0.5, j + 0.5, u"\u2190", va='center', ha='center')
            

plt.matshow(V_matrix,cmap='jet')
plt.colorbar()
ax.set_xlim(min_val,max_val)
ax.set_ylim(max_val,min_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(np.arange(max_val))
ax.grid()
        
        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
fig, ax = plt.subplots()

min_val, max_val = 0, 10

intersection_matrix = np.zeros(shape=(10,10))
intersection_matrix[0,0] = 10
for i in range(10):
    for j in range(10):
        c = intersection_matrix[i][j]
        ax.text(i + 0.5, j + 0.5, str(c), va='center', ha='center')

plt.matshow(intersection_matrix,cmap='jet')
#plt.show()
ax.set_xlim(min_val, max_val)
ax.set_ylim(max_val, min_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(list(np.arange(max_val))[::-1])
#ax = plt.axis()
#plt.axis((ax[0],ax[1], ax[3], ax[2]))
ax.grid()
            
          
x = np.random.randn(1000)
y = np.random.randn(1000)+5
plt.hist2d(x, y, bins=40, cmap='jet')
plt.colorbar()


fig, ax = plt.subplots()

min_val, max_val, diff = 0., 10., 1.

#imshow portion
N_points = int((max_val - min_val) / diff)
imshow_data = np.random.rand(N_points, N_points)
ax.imshow(imshow_data, interpolation='nearest')

#text portion
ind_array = np.arange(min_val, max_val, diff)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = 'x' if (x_val + y_val)%2 else 'o'
    ax.text(x_val, y_val, c, va='center', ha='center')

#set tick marks for grid
ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(min_val-diff/2, max_val-diff/2)
ax.set_ylim(min_val-diff/2, max_val-diff/2)
ax.grid()
plt.show()

#reward function 2
    
#create a 3d tensor
#4 by 4 by 100

epsilon = 0.01
V2 = [0] * 100
delta = math.inf
discount_factor = 0.8

while delta > epsilon:
    delta = 0
    for s in range(100):
        v = V2[s]
        temp = []
        for action in range(4):
            sums = 0
            for s_prime in range(100):
                sums += transition_probabilities_matrix[action][s][s_prime] * (reward_2.flatten(order='F')[s_prime] + discount_factor * V2[s_prime])
            
            temp.append(sums)
        
        V2[s] = max(temp)
        delta = max([delta, abs(v - V2[s])])
        

best_action_2 = []
for s in range(100):
    temp = []
    for action in range(4):
        sums = 0
        for s_prime in range(100):
            sums += transition_probabilities_matrix[action][s][s_prime] * (reward_2.flatten(order='F')[s_prime] + discount_factor * V2[s_prime])
            
        temp.append(sums)
    
    best_action_2.append(np.argmax(temp))
    

V2_matrix = np.array(V2).reshape((10,10))

fig,ax = plt.subplots()
for i in range(10):
    for j in range(10):
        c = V2_matrix[i][j]
        ax.text(i + 0.5, j + 0.5, str(round(c,2)), va='center',ha='center')

plt.matshow(V2_matrix,cmap='jet')
plt.colorbar()
ax.set_xlim(min_val,max_val)
ax.set_ylim(max_val,min_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(np.arange(max_val))
ax.grid()



best_action_matrix_2 = np.array(best_action_2).reshape((10,10))
fig,ax = plt.subplots()
for i in range(10):
    for j in range(10):
        c = best_action_matrix_2[i][j]
        action = actions[c]
        
        
        if action == 'up':
            ax.text(i + 0.5, j + 0.5, u'\u2191', va='center', ha='center')
        elif action == 'down':
            ax.text(i + 0.5, j + 0.5, u'\u2193', va='center', ha='center')
        elif action == 'right':
            ax.text(i + 0.5, j + 0.5, u"\u2192", va='center', ha='center')
        else:
            ax.text(i + 0.5, j + 0.5, u"\u2190", va='center', ha='center')
            

plt.matshow(V_matrix,cmap='jet')
plt.colorbar()
ax.set_xlim(min_val,max_val)
ax.set_ylim(max_val,min_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(np.arange(max_val))
ax.grid()
        