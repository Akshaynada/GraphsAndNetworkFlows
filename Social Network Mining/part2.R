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
