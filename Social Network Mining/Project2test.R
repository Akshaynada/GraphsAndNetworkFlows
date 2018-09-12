
library("igraph")
#dd = read.table("facebook_combined.txt")
facebook_graph= read.graph("facebook_combined.txt",directed=FALSE)
facebook_graph$names = V(facebook_graph)
is_connected(facebook_graph)

diameter(facebook_graph)


facebook_graph_degree = degree(facebook_graph)

plot(degree_distribution(facebook_graph))

plot(degree_distribution(facebook_graph), log='xy')


xmax <- max(degree(facebook_graph))
degree_range <- as.numeric(1:xmax)
facebook_dd = degree_distribution(facebook_graph)

degree_dist <- degree_distribution(facebook_graph)[degree_range]
nonzero.position <- which(facebook_dd > 0)
degree_dist <- degree_dist[nonzero.position]
degree_range <- degree_range[nonzero.position]

reg <- lm(log(degree_dist) ~ log(degree_range))
coefficients <- coef(reg)
func_powerlaw <- function(x) exp(coefficients[[1]] + coefficients[[2]] * log(x))
alpha <- -coefficients[[2]]

plot(degree_dist ~ degree_range, log='xy')
curve(func_powerlaw, col='red', add=TRUE, from=10,to=500,n=length(degree_range))

# y = mx + b
print(xmin)
curve(func_powerlaw, col='red', add=TRUE, to=xmax,from=xmin,n=length(degree_range))


#function to create personal network given a core node
create_personal_network = function(graph,core_node)
{
  core_neighbors = neighborhood(graph,order=1,nodes=core_node)
  personal_network = induced_subgraph(graph,unlist(core_neighbors))
  personal_network$names = sort(unlist(core_neighbors))
  return(personal_network)
}

neighbors = ego(facebook_graph,nodes=c(1))
node1_personal_network = create_personal_network(facebook_graph,1)
plot(node1_personal_network,vertex.size=0,vertex.label=NA)

core.nodes = c()
core.node.degrees = c()
for(i in 1:vcount(facebook_graph))
{
  s = degree(facebook_graph,v=c(i))
  if(s > 200) {
    core.nodes = c(core.nodes, i)
    core.node.degrees = c(core.node.degrees,degree(facebook_graph,v=c(i)))
  }
}

print(mean(core.node.degrees))


#1.3.1
titles = c("Node 1", "Node 108", "Node 349", "Node 484", "Node 1087")
#core_subgraphs = make_ego_graph(facebook_graph,nodes=c(1,108,349,484,1087))

nodes = c(1,108,349,484,1087)
for(i in 1:length(nodes)){
  g = create_personal_network(facebook_graph, nodes[i])
  c1 = fastgreedy.community(g)
  
  print(paste("Community sizes for Fast Greedy ", i))
  print(sizes(c1))
  colors = rainbow(max(membership(c1)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  nodeSizes = rep(4,length(c1$membership))
  V(g)$color = nodes_colors[membership(c1)]
  nodeSizes[which(g$names == nodes[i])] = 8
  m1 = modularity(c1)
  print(paste("Modularity of ", titles[i], " using Fast Greedy: ", m1))
  plot(g,vertex.size=nodeSizes,vertex.label=NA,edge.color='grey', main=paste("Community Structure using Fast Greedy",titles[i]))
  dev.copy(png, paste(titles[i],'FastGreedy.png'))
  dev.off()
  c2 = cluster_edge_betweenness(g)
  print(paste("Community sizes for EdgeBetweeness ", i))
  print(sizes(c2))
  
  colors = rainbow(max(membership(c2)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  nodeSizes = rep(4,length(c2$membership))
  V(g)$color = nodes_colors[membership(c2)]
  nodeSizes[which(g$names == nodes[i])] = 8
  m2 = modularity(c2)
  print(paste("Modularity of ", titles[i], " using Edge Betweeness: ", m2))
  plot(g,vertex.size=nodeSizes,vertex.label=NA,edge.color='grey',main=paste("Community Structure using Edge Betweeness", titles[i]))
  dev.copy(png, paste(titles[i],'EdgeBetweeness.png'))
  dev.off()
  c3 = cluster_infomap(g)
  print(paste("Community sizes for Infomap ", i))
  print(sizes(c3))
  
  colors = rainbow(max(membership(c3)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  nodeSizes = rep(4,length(c3$membership))
  V(g)$color = nodes_colors[membership(c3)]
  nodeSizes[which(g$names == nodes[i])] = 8
  m3 = modularity(c3)
  print(paste("Modularity of ", titles[i], " using Infomap: ", m3 ))
  plot(g,vertex.size=nodeSizes,vertex.label=NA,edge.color='grey',main=paste("Community Structure using Infomap", titles[i]))
  dev.copy(png, paste(titles[i],'Infomap.png'))
  dev.off()
}


#1.3.2
nodeIds = c(1,108,349,484,1087)
for(i in 1:length(nodes)){
  g = create_personal_network(facebook_graph, nodes[i])
  pn_with_core_node_removed = delete.vertices(g, V(g)[which(g$names == i)])
  c1 = cluster_fast_greedy(pn_with_core_node_removed)
  print(paste("Community sizes for Fast Greedy Node Removed ", nodeIds[i]))
  print(sizes(c1))
  colors = rainbow(max(membership(c1)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(g)$color = nodes_colors[membership(c1)]
  m1 = modularity(c1)
  print(paste("Modularity of ", titles[i], " using Fast Greedy: ", m1))
  plot(g,vertex.size=4,vertex.label=NA, edge.color='grey', main=paste("Community Structure using Fast Greedy With Core Node ",nodeIds[i], " Removed"))
  dev.copy(png, paste(titles[i],'FastGreedyCoreRemoved.png'))
  dev.off()
  c2 = cluster_edge_betweenness(pn_with_core_node_removed)
  print(paste("Community sizes for EdgeBetweeness Node Removed ", nodeIds[i]))
  print(sizes(c2))
  colors = rainbow(max(membership(c2)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(g)$color = nodes_colors[membership(c2)]
  m2 = modularity(c2)
  print(paste("Modularity of ", titles[i], " using Edge Betweeness: ", m2))
  plot(g,vertex.size=4,vertex.label=NA,edge.color='grey',main=paste("Community Structure using Edge Betweeness With Core Node ", nodeIds[i], " Removed"))
  dev.copy(png, paste(titles[i],'EdgeBetweenessCoreRemoved.png'))
  dev.off()
  c3 = cluster_infomap(pn_with_core_node_removed)
  print(paste("Community sizes for InfoMap Node removed ", nodeIds[i]))
  print(sizes(c3))
  colors = rainbow(max(membership(c3)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(g)$color = nodes_colors[membership(c3)]
  m3 = modularity(c3)
  print(paste("Modularity of ", titles[i], " using Infomap: ", m3 ))
  plot(g,vertex.size=4,vertex.label=NA,edge.color='grey',main=paste("Community Structure using Infomap With Core Node ", nodeIds[i], " Removed"))
  dev.copy(png, paste(titles[i],'InfomapCoreRemoved.png'))
  dev.off()
}


#1.3.3


#embeddedness of node is equal to the degree of the node
#function to get mutual friends between two nodes
mutual_friends = function(graph,i,j) {
  neighborsOfI = neighbors(graph,V(graph)[which(graph$names == i)])
  neighborsOfJ = neighbors(graph,V(graph)[which(graph$names == j)])
  mutualFriends = intersect(neighborsOfI,neighborsOfJ)
  return(mutualFriends)
}

#function to get embeddedness between two nodes
get_embeddedness = function(graph,i,j){
  return(length(mutual_friends(graph,i,j)))
}

#function to get dispersion between two nodes
get_dispersion = function(graph,i,j){
  mutualFriends = mutual_friends(graph,i,j)
  pruned_subgraph = delete_vertices(graph, c(which(graph$names == i), which(graph$names == j)))

  pruned_subgraph$names = sort(personalNetwork$names[intersect(which(graph$names != i), which(graph$names !=  j))])
  
  dispersion_measure = numeric(0)
  #306 and 328
  if(length(mutualFriends) < 2) {
    dispersion_measure = c(0)
  } else {
    
    possible_pairs = combn(mutualFriends,2)
    for(i in 1:(length(possible_pairs)/2)) {
      #print(paste("pair 1", possible_pairs[,i][1], " pair 2", possible_pairs[,i][2]))

      first_node = V(pruned_subgraph)[which(pruned_subgraph$names == graph$names[which(V(graph) == possible_pairs[,i][1])])]
      #print(paste("Which index equals ", which(pruned_subgraph$names == graph$names[which(V(graph) == possible_pairs[,i][1])])))
      #print(paste("Which FirstNode: " , first_node))
      second_node = V(pruned_subgraph)[which(pruned_subgraph$names == graph$names[which(V(graph) ==possible_pairs[,i][2])])]
      #print(paste("Which second index equals ", which(pruned_subgraph$names == graph$names[which(V(graph) == possible_pairs[,i][2])])))
      #print(paste("Second node: ", second_node))
      #if((first_node %in% neighbors(pruned_subgraph,second_node)) == FALSE ) {
        #first_node_neighbors = neighbors(pruned_subgraph,first_node)
        #second_node_neighbors = neighbors(pruned_subgraph,second_node)
        #if(length(intersect(first_node_neighbors,second_node_neighbors)) == 0) {
          #dispersion_measure = c(dispersion_measure,1)
        #} else {
          #dispersion_measure = c(dispersion_measure, 0)
        #}
      #} else{
        #dispersion_measure = c(dispersion_measure,0)
      #}
      dispersion_measure = c(dispersion_measure, shortest.paths(pruned_subgraph, first_node, second_node))
    }
    
    
  }
  
  return(sum(dispersion_measure))
}


embeddedness = numeric(0)
dispersion = numeric(0)


for(i in nodeIds) {
  personalNetwork = create_personal_network(facebook_graph,i)
  
  embeddedness = numeric(0)
  dispersion = numeric(0)
  for(j in personalNetwork$names) {
    if(j == i)
    {
      next #skip if same node
    }
      
      
    embeddedness = c(embeddedness,get_embeddedness(personalNetwork,i,j))
    dispersion = c(dispersion, get_dispersion(personalNetwork,i,j))
    
  }
  
  dispersion[mapply(is.infinite,dispersion)] = 0
  #108 and 172
  hist(embeddedness,main=paste("Embeddedness Distribution for Core Node ", i), col='blue')
  
  dev.copy(png, paste(i,'-Embeddedness-Distribution.png'))
  dev.off()
  hist(dispersion,main=paste("Dispersion Distribution for Core Node ", i),col='blue')
  dev.copy(png, paste(i,'-Dispersion-Distribution.png'))
  dev.off()
  
  #dispersion[mapply(is.infinite,dispersion)] = 0
  ratio = dispersion / embeddedness
  max_embeddedness = personalNetwork$names[which(embeddedness == max(embeddedness))]
  max_dispersion = personalNetwork$names[which(dispersion == max(dispersion))]
  
  ratio[mapply(is.nan,ratio)] = 0
  
  max_ratio = personalNetwork$names[which(ratio == max(ratio))]
  
  #get community structure using fastgreedy method
  cs = fastgreedy.community(personalNetwork)
  
  max_embedded_node = which(personalNetwork$names == max_embeddedness)
  core_node = which(personalNetwork$names == i)
  max_dispersion_node = which(personalNetwork$names == max_dispersion)
  max_ratio_node = which(personalNetwork$names == max_ratio)
  
  
  #nodeColors = cs$membership + 1
  nodeSizes = rep(2,length(cs$membership))
  edgeColors = rep("grey", length(E(personalNetwork)))
  edgeWeights = rep(0.25, length(E(personalNetwork)))
  
  edgeColors[which(get.edgelist(personalNetwork,names=F)[,1] == max_dispersion_node | get.edgelist(personalNetwork,names=F)[,2] == max_dispersion_node)] = 'red'
  edgeWeights[which(get.edgelist(personalNetwork,names=F)[,1] == max_dispersion_node | get.edgelist(personalNetwork,names=F)[,2] == max_dispersion_node)] = 6
  
  
  colors = rainbow(max(membership(cs)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(personalNetwork)$color = nodes_colors[membership(cs)]
  V(personalNetwork)[max_dispersion_node]$color = 'gold'
  V(personalNetwork)[core_node]$color = 'black'
  
  nodeSizes[max_dispersion_node] = 8
  #nodeColors[max_dispersion_node] = 7
  nodeSizes[core_node] = 4
  #nodeColors[core_node] = 
  
  #co = layout_with_fr(g,niter=1000)
  plot(personalNetwork,vertex.size=nodeSizes,vertex.label=NA,edge.width=edgeWeights,edge.color=edgeColors,main=paste("Node ", i, " Community Structure with Max Dispersion Node(In Gold)"), layout=layout.fruchterman.reingold(personalNetwork))
  dev.copy(png, paste("Node",i,'MaxDispersionNode.png'))
  dev.off()
  
  #plot max embedded node
  
  nodeSizes = rep(2,length(cs$membership))
  edgeColors = rep("grey", length(E(personalNetwork)))
  edgeWeights = rep(0.25, length(E(personalNetwork)))
  edgeColors[which(get.edgelist(personalNetwork,names=F)[,1] == max_embedded_node | get.edgelist(personalNetwork,names=F)[,2] == max_embedded_node)] = 'red'
  edgeWeights[which(get.edgelist(personalNetwork,names=F)[,1] == max_embedded_node | get.edgelist(personalNetwork,names=F)[,2] == max_embedded_node)] = 6
  
  
  colors = rainbow(max(membership(cs)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(personalNetwork)$color = nodes_colors[membership(cs)]
  V(personalNetwork)[max_embedded_node]$color = 'gold'
  V(personalNetwork)[core_node]$color = 'black'
  
  nodeSizes[max_embedded_node] = 8
  #nodeColors[max_dispersion_node] = 7
  nodeSizes[core_node] = 4
  #nodeColors[core_node] = 
  
  plot(personalNetwork,vertex.size=nodeSizes,vertex.label=NA,edge.width=edgeWeights,edge.color=edgeColors,main=paste("Node ", i, " with Max Embeddedness Node"),layout=layout.fruchterman.reingold(personalNetwork))
  dev.copy(png, paste("Node",i,'MaxEmbeddedednessNode.png'))
  dev.off()
  
  
  #plot max ratio
  nodeSizes = rep(2,length(cs$membership))
  edgeColors = rep("grey", length(E(personalNetwork)))
  edgeWeights = rep(0.25, length(E(personalNetwork)))
  edgeColors[which(get.edgelist(personalNetwork,names=F)[,1] == max_ratio_node | get.edgelist(personalNetwork,names=F)[,2] == max_ratio_node)] = 'red'
  edgeWeights[which(get.edgelist(personalNetwork,names=F)[,1] == max_ratio_node | get.edgelist(personalNetwork,names=F)[,2] == max_ratio_node)] = 6
  
  
  colors = rainbow(max(membership(cs)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  V(personalNetwork)$color = nodes_colors[membership(cs)]
  V(personalNetwork)[max_ratio_node]$color = 'gold'
  V(personalNetwork)[core_node]$color = 'black'
  
  nodeSizes[max_ratio_node] = 8
  #nodeColors[max_dispersion_node] = 7
  nodeSizes[core_node] = 4
  #nodeColors[core_node] = 
  
  plot(personalNetwork,vertex.size=nodeSizes,vertex.label=NA,edge.width=edgeWeights,edge.color=edgeColors,main=paste("Node ", i, " Community Structure with Max Ratio Node"),layout=layout.fruchterman.reingold(personalNetwork))
  dev.copy(png, paste("Node",i,'MaxRatioNode.png'))
  dev.off()
  
  
  
  
}

#######################################1.4#######################################################################


###### 1.4.2

#Go through every user in network
for(i in 1:vcount(facebook_graph))
{
  nodes = c()
  jaccard_measure = c() #store jacccard measures for node i
  neighbors_of_i = neighbors(facebook_graph,i)
  
  for(j in 1:vcount(facebook_graph))
  {
    if(j %in% neighbors_of_i == TRUE)
    {
      next
    }
    neighbors_of_j = neighbors(facebook_graph,j)
    jaccard_measure_between_i_and_j = length(intersect(neighbors_of_i, neighbors_j)) / length(union(neighbors_of_i,neighbors_of_j))
    jaccard_measure = c(jaccard_measure, jaccard_measure_between_i_and_j)
    nodes = c(nodes,j)
    
    
  }
  
  indexes = order(jaccard_measure,decreasing=TRUE) #sort in descending order
  top_t_friends = nodes[indexes[1:t]] #get top t jaccard measure scores
  
}


###### 1.4.3 Creating the list of users

node_415_personal_network = create_personal_network(facebook_graph,415)

nodes_with_degree_24 = c()

for(i in node_415_personal_network$names)
{
  if(degree(node_415_personal_network, V(node_415_personal_network)[which(node_415_personal_network$names == i)]) == 24)
  {
    nodes_with_degree_24 = c(nodes_with_degree_24,i)
  }
}


print(paste("Length of Nodes with Degree 24 ", length(nodes_with_degree_24)))


#1.4.4 Average accuracy of friend recommendation algorithm

total_jaccard_accuracies = c()
total_common_neighbor_accuracies = c()
total_adamic_adar_accuracies = c()



for(i in nodes_with_degree_24)
{
  
  
  
  common_neighbor_accuracies = c()
  jaccard_accuracies = c()
  adamic_adar_accuracies = c()
  
  
  
  for(n in 1:10)
  {
    temp_graph = node_415_personal_network
    true_index = V(temp_graph)[which(node_415_personal_network$names == i)]
    incident_edges = E(temp_graph)[to(true_index)]
    number_of_indicent_edges = length(incident_edges)
    edges_to_delete = c()
    deleted_friends = c()
    
    for(j in 1:length(E(temp_graph)))
    {
      
      
      edge = E(temp_graph)[j]
      
      if(edge %in% incident_edges)
      {
        random_number = runif(1)
        if(random_number <= 0.25)
        {
          
          edges_to_delete = c(edges_to_delete,j)
          endpoints = ends(temp_graph, edge)
          
          #temp_graph = delete_edges(temp_graph,edge)
          #temp_graph = temp_graph - edge
          if(endpoints[1,1] != V(temp_graph)[which(node_415_personal_network$names == i)])
          {
            deleted_friends = c(deleted_friends, endpoints[1,1])
          }
          
          if(endpoints[1,2] != V(temp_graph)[which(node_415_personal_network$names == i)])
          {
            deleted_friends = c(deleted_friends,endpoints[1,2])
          }
        } 
      }
      
      
    }
    
    temp_graph = delete_edges(temp_graph,edges_to_delete)
    
    number_of_friends_to_recommend = length(deleted_friends)
    
    nodes = c()
    jaccard_measure = c() #store jacccard measures for node i
    common_neighbors_measure = c()
    adamic_adar_measure = c()
    neighbors_of_i = neighbors(temp_graph,V(temp_graph)[which(node_415_personal_network$names == i)])
    
    for(j in 1:vcount(temp_graph))
    {
      if(((j %in% neighbors_of_i) == TRUE) || j == true_index)
      {
        next #skip if j is a neighbor of i or j equals i
      }
      neighbors_of_j = neighbors(temp_graph,j)
      jaccard_measure_between_i_and_j = length(intersect(neighbors_of_i, neighbors_of_j)) / length(union(neighbors_of_i,neighbors_of_j))
      common_neighbors_measure_between_i_and_j = length(intersect(neighbors_of_i, neighbors_of_j))
      
      adamic_adar_values = c()
      for(k in intersect(neighbors_of_i,neighbors_of_j))
      {
        neighbors_of_k = neighbors(temp_graph,k)
        adamic_adar_values = c(adamic_adar_values,1/log(length(neighbors_of_k)))
      }
      
      adamic_adar_measure = c(adamic_adar_measure, sum(adamic_adar_values))
      jaccard_measure = c(jaccard_measure, jaccard_measure_between_i_and_j)
      common_neighbors_measure = c(common_neighbors_measure,common_neighbors_measure_between_i_and_j)
      nodes = c(nodes,j)
      
      
    }
    
    jaccard_indexes = order(jaccard_measure,decreasing=TRUE) #sort in descending order
    common_neighbor_indexes = order(common_neighbors_measure,decreasing=TRUE)
    adamic_adar_indexes = order(adamic_adar_measure,decreasing=TRUE)
    
    
    top_jaccard_friends = nodes[jaccard_indexes[1:number_of_friends_to_recommend]] #get top t jaccard measure scores
    top_common_neighbor_friends = nodes[common_neighbor_indexes[1:number_of_friends_to_recommend]]
    top_adamic_adar_friends = nodes[adamic_adar_indexes[1:number_of_friends_to_recommend]]
    
    jaccard_accuracy = length(intersect(top_jaccard_friends, deleted_friends)) / length(deleted_friends)
    common_neighbor_accuracy = length(intersect(top_common_neighbor_friends, deleted_friends)) / length(deleted_friends)
    adamic_adar_accuracy = length(intersect(top_adamic_adar_friends, deleted_friends)) / length(deleted_friends)
    
    common_neighbor_accuracies = c(common_neighbor_accuracies,common_neighbor_accuracy)
    jaccard_accuracies = c(jaccard_accuracies, jaccard_accuracy)
    adamic_adar_accuracies = c(adamic_adar_accuracies, adamic_adar_accuracy)
  
  
  }
  
  total_jaccard_accuracies = c(total_jaccard_accuracies, mean(jaccard_accuracies))
  total_common_neighbor_accuracies = c(total_common_neighbor_accuracies, mean(common_neighbor_accuracies))
  total_adamic_adar_accuracies = c(total_adamic_adar_accuracies, mean(adamic_adar_accuracies))
  
  
}


accuracy_for_jaccard = mean(total_jaccard_accuracies)
accuracy_for_common_neighbor = mean(total_common_neighbor_accuracies)
accuracy_for_adamic_adar = mean(total_adamic_adar_accuracies)

print(paste("Accuracy for Jaccard Measure: ", accuracy_for_jaccard))
print(paste("Accuracy for Common Neighbor Measure: ", accuracy_for_common_neighbor))
print(paste("Accuracy for Adamic Adar Measure: ", accuracy_for_adamic_adar))



######## PART 2 ######################
library('igraph')



# read the google+ ego networks
file_names = list.files("gplus/")
file_ids = sub("^([^.]*).*", "\\1", file_names)
ego_node_ids = unique(file_ids)



cat("Total Number of Ego Nodes = ", length(ego_node_ids))

ids_circles = numeric()


for (id in ego_node_ids) {
  # get the number of circles
  circles_file = paste("gplus/" , id , ".circles" , sep="")
  circles_connect = file(circles_file , open="r")
  circles_content = readLines(circles_connect)
  close(circles_connect)
  
  # check if greater than 2
  if(length(circles_content) > 2)
    ids_circles = c(ids_circles, id)
}


cat("Total IDs with > 2 circles = ", length(ids_circles))

circle_person_ids <- c('109327480479767108490','115625564993990145546','101373961279443806744')

for (id in  circle_person_ids) {
  
  edges_file = paste("gplus/" , id  , ".edges" , sep="") # edge list
  circles_file = paste("gplus/" , id , ".circles" , sep="") # circles list
  
  circles_connect = file(circles_file , open="r")
  circles_content = readLines(circles_connect)
  
  circles = list()
  
  
  numberInCircle = c()
  for (i in 1:length(circles_content)) {
    circle_nodes = unlist(strsplit(circles_content[i],"\t"))
    circle_nodes = circle_nodes[2: length(circle_nodes)]
    
    numberInCircle = c(numberInCircle, length(circle_nodes))
    #circle_nodes = circle_nodes[1: length(circle_nodes)]
    # print(length(unique(circle_nodes)))p
    # all_nodes = c()
    # for (j in 1:length(circle_nodes)) {
    #   all_nodes = c(all_nodes, circle_nodes[j])
    # }
    # print(all_nodes)
    # print(length(all_nodes[1]))
    # print(length(unique(all_nodes[1])))
    circles = c(circles, circle_nodes)
  }
  
  
  circleInformationNodes = (unlist(circles))
  N = length(unique(unlist(circles)))
  print(N)
  close(circles_connect)
  
  
  
  # create network using edge list
  g_network = read.graph(edges_file , format = "ncol" , directed=TRUE)  
  g_network = add.vertices(g_network, nv = 1, name = id)
  ego_node_index = which(V(g_network)$name==id) 
  
  add_edge_list = c()
  for (vertex in 1:(vcount(g_network) - 1)) {
    add_edge_list = c(add_edge_list, c(ego_node_index, vertex))
  }
  
  g_network = add_edges(g_network, add_edge_list)
  
  #plot(g_network,vertex.label = NA, vertex.size = 4);
  walktrap_comm = walktrap.community(g_network)
  b = c()
  for(i in 1:length(sizes(walktrap_comm))) {
    nodesInCommunity = names(membership(walktrap_comm)[membership(walktrap_comm) == i])
    nodesInCommunityWithCircleInfo = intersect(nodesInCommunity,circleInformationNodes)
    b = c(b, length(nodesInCommunityWithCircleInfo))
  }
  
  cij = matrix(nrow=length(walktrap_comm),ncol=length(circles_content))
  for(i in 1:length(walktrap_comm)) {
    nodesInCommunity = names(membership(walktrap_comm)[membership(walktrap_comm) == i])
    for(j in 1:length(circles_content)) {
      nodesInCircle = unlist(strsplit(circles_content[j],"\t"))
      cij[i,j] = length(intersect(nodesInCommunity,nodesInCircle))
    }
      
  }
  
  #Calculate homogoneity
  
  h.c = 0
  h.k = 0
  for(i in 1:length(circles_content))
  {
    h.c = h.c + (numberInCircle[i] / N) * (log(numberInCircle[i] / N))
  }
  
  for(i in 1:length(walktrap_comm))
  {
    if(b[i] != 0)
      h.k = h.k + (b[i] / N) * (log(b[i] / N))
  }
  
  h.c = -h.c
  h.k = -h.k
  
  h.c.given.k = 0
  h.k.given.c = 0
  for(j in 1:length(walktrap_comm)) {
    for(i in 1:length(circles_content)) {
      if(cij[j,i] != 0)
        h.c.given.k = h.c.given.k + (cij[j,i] / N) * (log(cij[j,i] / b[j]))
      
    }
  }
  
  h.c.given.k = -h.c.given.k
  for(i in 1:length(circles_content)) {
    for(j in 1:length(walktrap_comm)) {
      if(cij[j,i] != 0){
        h.k.given.c = h.k.given.c + (cij[j,i] / N) * (log(cij[j,i] / numberInCircle[i]))
      }
      
    }
  }
  
  h.k.given.c = -h.k.given.c
  
  h = 1 - ((h.c.given.k) / h.c)
  
  c = 1 - ((h.k.given.c) / h.k)
  
  print(paste("h ", h))
  print(paste("c ", c))
  modularity_comm = modularity(walktrap_comm);
  
  cat("The modulariity is ", modularity_comm)
  colors = rainbow(max(membership(walktrap_comm)) + 1)
  nodes_colors = colors[1:(length(colors) - 1)]
  nodeSizes = rep(4,length(walktrap_comm$membership))
  V(g_network)$color = nodes_colors[membership(walktrap_comm)]
  #plot(walktrap_comm,g_network)
  
  #plot(degree.distribution(g_network, mode="in"),main=paste("In degree distribution of node ",id), xlab='degree',ylab='frequency')
  # plot(degree.distribution(g_network, mode="out"),main=paste("Out degree distribution",id), xlab='degree',ylab='frequency')
  
  
  
}
