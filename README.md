# GraphsAndNetworkFlows
EE232 ( Graphs and Network Flows ) Projects

Team Members 
1. Akshay Shetty
2. Zongheng Ma
3. Cyrus Tabatabai-Yazdi
4. Bakari Hassan

------------------------------------------

1. Social Network Mining 

a. Facebook dataset 

i) Study on the structural properties of Facebook Network such as connectivity and degree deistribution.
ii)A personal (ego) network was generated for a specific node. This personalized network was limited to all neighbors within 1 degree of the ego node. All connections between two nodes that are both within the network are maintained.
iii)Focuses on on the social characteristics of core nodes’ personalized networks. This includes community structure, modularity, embeddedness, and dispersion. These characteristics and measures will allow us to make inferences regarding interpersonal relationships between users.
iv) Three algorithms for friend recommendation were explored. First, an ego graph was generated for node 415. Then, all of node 415’s friends with 24 degrees were selected. For each node, its friends were randomly deleted with probability 0.25 and then replaced with new friends within the ego network based on the chosen similarity measure. The measures explored include Adamic Adar, Jaccard, and Common Neighbors.


b.Google Plus dataset 
The Google+ network is explored and analyzed to gain an understanding of its community structure. Additionally, network homogeneity and completeness measures (two commonly used measures for clustering performance assessment) were implemented in the context of network graphs.


2. IMDB Data Mining

a. Our goal was to explore the properties of actors and actresses by constructing a directed network in order to capture and analyze relationships.
b. Create an undirected movie network to analyze the relationships between various movies

3. Reinforcement and Inverse Reinforcement Learning

a. The goal of Part 1 was to train an agent to navigate in a gridworld using an optimal policy derived using the value iteration algorithm. The agent starts from the top left of a grid and has to navigate to the bottom right. Each state has a reward associated with it and our goal is to calculate the values of being in each possible state and the optimal action to take at each state
b. The goal of part 2 is to infer a reward function by learning from an expert. In our case, the “expert” is the agent following the optimal policy for reward function’s 1 and 2 respectively. Thus, we want the agent to learn a good reward function that can lead to as best of an optimal policy as possible.
