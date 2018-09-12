# -*- coding: utf-8 -*-
"""
Created on Fri May 18 20:42:28 2018

@author: cyrus
"""
import numpy as np
from cvxopt import matrix,solvers
import matplotlib.pyplot as plt

def normalize(vals):
  
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)
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
best_actions = [0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 3,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 3,
 3,
 3,
 0,
 0,
 0,
 0,
 0,
 0,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 0,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0]

best_actions_2 = [3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 3,
 3,
 3,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 3,
 3,
 3,
 0,
 0,
 2,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0,
 0,
 0,
 0,
 3,
 3,
 3,
 1,
 1,
 3,
 0,
 0,
 0,
 0,
 2,
 3,
 1,
 1,
 3,
 3,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 3,
 3,
 0,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 0]

rewards_1 = np.zeros((100))
rewards_1[99] = 1

def IRL(optimal, lmda, prob, b_val):
    D = np.zeros((1000, 3 * 100))
    b = np.zeros((1000))
    c = np.zeros((300))
    
    discount = 0.8
    
    for j in range(100):
        optimal_action = optimal[j]
        inv = np.linalg.inv(np.identity(100) - discount * prob[optimal_action,:,:])
        
        temp = 0
        for a in range(4):
            if a != optimal_action:
                D[j * 3 + temp, :100] = -np.dot(prob[optimal_action,j,:] - transition_probabilities_matrix[a,j,:], inv)
                D[300 + j * 3 + temp,:100] = -np.dot(prob[optimal_action,j,:] - transition_probabilities_matrix[a,j,:], inv)
                D[300 + j * 3 + temp, 100 + j] = -1
                temp += 1
    
    
    for j in range(100):
        D[600 + j, j] = 1
        b[600+j]=b_val
    
    
    for j in range(100):
        D[700 + j, j] = -1
        b[700+j]=b_val
    
    for j in range(100):
        D[800 + j, j] = 1
        D[800 + j, 200 + j] = -1
    
    for j in range(100):
        D[900 + j, j] = -1
        D[900 + j, 200 + j] = -1
#        b[900+j]=b_val
    
    
    for j in range(100):
        c[100:200] = -1
        c[200:] = lmda
    
    solution = solvers.lp(matrix(c), matrix(D), matrix(b))
    
    res = solution['x'][:100]
    
    return res


#question 11
lambdas = np.linspace(0,5,500)

accuracies = []
for i in range(len(lambdas)):
    
    current_lambda = lambdas[i]
    
    reward = IRL(best_actions, current_lambda, transition_probabilities_matrix, 1)
    
    epsilon = 0.01
    V = [0] * 100
    delta = 10**12
    discount_factor = 0.8

    while delta > epsilon:
        delta = 0
        for s in range(100):
            v = V[s]
            temp = []
            for action in range(4):
                sums = 0
                for s_prime in range(100):
                    sums += transition_probabilities_matrix[action][s][s_prime] * (reward[s_prime] + discount_factor * V[s_prime])
            
                temp.append(sums)
        
            V[s] = max(temp)
            delta = max([delta, abs(v - V[s])])
        

    best_action = []
    for s in range(100):
        temp = []
        for action in range(4):
            sums = 0
            for s_prime in range(100):
                sums += transition_probabilities_matrix[action][s][s_prime] * (reward[s_prime] + discount_factor * V[s_prime])
            
            temp.append(sums)
    
        best_action.append(np.argmax(temp))
    
    s = 0
    for i,action in enumerate(best_action):
        s += 1 if best_actions[i] == action else 0
    
    accuracies.append(s / 100)

plt.title("Lambda vs Accuracy for Reward Function 1")
plt.plot(lambdas, accuracies)

#question 12
dummy = []
for i in range(len(lambdas)):
    dummy.append((lambdas[i], accuracies[i]))    
dummy.sort(key = lambda x: x[1])
print(dummy[-1])

#question 13
extracted_reward = np.array(IRL(best_actions, 4.6192384769539077, transition_probabilities_matrix, 1)).reshape((10, 10))
plt.imshow(extracted_reward,cmap=plt.cm.Reds)
plt.colorbar()
