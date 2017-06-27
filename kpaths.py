from __future__ import division

import pcst_fast as pc

# import collections
# import math 
import networkx as nx
import sys
import folium
import webbrowser
import os
import glob
import csv


# print graph edges
def printEdges(Graph):
    edge_attrs = nx.get_edge_attributes(Graph, "weight")
    for edge,weight in edge_attrs.items():
        num = Graph.edge[edge[0]][edge[1]]['num']
        print "Edge: ",edge," with cost: ",weight, " with number of active components: ",num# ," and is active: ",active


# print graph vertices
def printVertices(Graph):
    node_attrs = nx.get_node_attributes(Graph, "prize")
    for node,weight in node_attrs.items():
        # active = Graph.node[node]['active']
        print "Node: ",node," with prize: ",weight# ," and is active: ",active

def getPrizes(OriginalGraph):
    node_prizes = {}
    for (key,value) in OriginalGraph.nodes(data=True):
        prize = value['prize']
        node_prizes[key] = prize
    
    return node_prizes

def getTotalPrize(node_list,OriginalGraph,node_prizes):
    prize = 0;
    Graph = nx.Graph(OriginalGraph)
    remove = list(set(Graph.nodes(data=False)) - set(node_list))
    for n in remove:
        Graph.remove_node(n)
        
    for node in Graph.nodes(data=False):
        prize+=node_prizes[node]
        
    return prize

# can be used to plot points of trees on a map
def plotMap(points, file_out, intersection_coordinates):

    colors = ["#c30bd4", "#2096ba", "#e6b800","#5870ff", "#a70100", "#5da4ff",
              "#ff5461", "#00285c", "#ff2595", "#643f00", "#ff9dde", "#ffae9d",
              "#3333ff", "#339933", "#660033", "#3366cc", "#33cc33", "#ffcc66",
              "#66ffcc","#ff6666"]

    lats = []
    longs = []
    limit = len(points)
    
    for nds in points[0:limit]:
        nodes = []
        if isinstance(nds,int):
            nodes.append(nds)
        else:
            nodes = nds
            
        if len(nodes) > 1:
            for node in nodes:
                coord = intersection_coordinates[str(node)]
                lat_long = [x.strip() for x in coord.split(',')]
                lat = float(lat_long[0])
                lon = float(lat_long[1])
                lats.append(lat)
                longs.append(lon)
        else:
            coord = intersection_coordinates[str(nodes[0])]
            lat_long = [x.strip() for x in coord.split(',')]
            lat = float(lat_long[0])
            lon = float(lat_long[1])
            lats.append(lat); 
            longs.append(lon); 

    mean_lat = sum(lats) / len(lats)
    mean_lon = sum(longs) / len(longs)

    map_osm = folium.Map(location=[mean_lat, mean_lon], zoom_start=14,\
                        tiles = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy;<a href="https://carto.com/attribution">CARTO</a>')

    # add a marker for every record in the filtered data, use a clustered view
    for component, nds in enumerate(points[0:limit]):
        nodes = []
        if isinstance(nds,int):
            nodes.append(nds)
        else:
            nodes = nds
            
        for node in nodes:
            coord = intersection_coordinates[str(node)]
            lat_long = [x.strip() for x in coord.split(',')]
            lat = float(lat_long[0])
            lon = float(lat_long[1])
            popup = 'component: %d\t node: %s' % (component, str(node))
            node_color = colors[component % len(colors)]
            map_osm.circle_marker(location=[lat, lon], radius=5,
                                  line_color=node_color,
                                  popup=popup, fill_opacity=40,
                                  fill_color=node_color)

    map_osm.create_map(path=file_out)
    
def prepareFastPCST(LamdaGraph):
    prizes = []
    costs = []
    edge_list = []
    node_dictionary = {}
    index_dictionary = {}
    edges_dictionary = {}
    
    index = 0 
    for node in LamdaGraph.nodes(data = False):
        node_dictionary[node] = index
        index_dictionary[index] = node
        prizes.append((LamdaGraph.node[node]['prize']))
        index+=1
        
    index = 0    
    for edge in LamdaGraph.edges(data = False):
        new_edge = []
        u = edge[0]; v = edge[1];
        cost = LamdaGraph.edge[u][v]['weight']
        costs.append(cost)
        nd1 = node_dictionary[u]; nd2 = node_dictionary[v];
        new_edge.append(nd1); new_edge.append(nd2);
        edge_list.append(new_edge)
        edges_dictionary[index] = edge
        index+=1
        
    return index_dictionary, node_dictionary, edges_dictionary, edge_list, prizes, costs

def decodeFast(index_dictionary,edges_dictionary,vertices,edges):
    vertices = list(vertices)
    decoded_nodes = []
    edges = list(edges)
    decoded_edges = []
    for node in vertices:
        nd = index_dictionary[node]
        decoded_nodes.append(nd)

    for edge in edges:
        ed = edges_dictionary[edge]
        decoded_edges.append(ed)
        
    return decoded_nodes, decoded_edges

def computeObjectiveValueGraph(OrigGraph):
     
    obj_cost = 0; objective_value = 0
    tot_nodes = OrigGraph.nodes(data=False)
    for node in tot_nodes:
        objective_value+=OrigGraph.node[node]['prize']
   
     
    return objective_value

def createEdgesFromK_DPPath(paths):
    edges = []; final_edges = []; counter = 1
    # print 'Create edges from paths:',paths
    for path in paths:
        edges = []
        if len(path) == 1:
            final_edges.append(edges)
        else:
            while counter < len(path):
                u = path[counter-1]; v = path[counter]
                edge = (u,v)
                edges.append(edge)
                counter+=1
                
        final_edges.append(edges)
        counter = 1

    return final_edges

def prepareNodeEdgeDict(tree,LamdaGraph):
    orig_prizes = {}
    orig_edges = {}
    LamdaG = nx.Graph(LamdaGraph)
    for node in tree.nodes(data=False):
        orig_prizes[node] = LamdaG.node[node]['prize']
        
    for edge in tree.edges(data=False):
        u = edge[0]; v = edge[1]
        orig_edges[(u,v)] = LamdaGraph.edge[u][v]['weight']
        
    return orig_prizes,orig_edges

def create_edges_BFS(tree,root):
        
    bfs_edges = nx.bfs_edges(tree,root)
    
    return bfs_edges

def binarizeTree(dfs_edges):
    counter = 0
    Graph =nx.DiGraph()
    num_edges = len(list(dfs_edges))
    prev_parent = float("inf")
    cur_parent = float("inf")
    # use negative numbers for dummy nodes
    dummy = 0

    for i,edge in enumerate(dfs_edges):
        cur_parent = edge[0]
        child = edge[1]
        if cur_parent != prev_parent:
            if prev_parent < 0:
                if (i+1) < num_edges:
                    next_edge = dfs_edges[i+1]
                    next_parent = next_edge[0]
                    if cur_parent == next_parent:
                        dummy = prev_parent
                        dummy = dummy - 1
                        Graph.add_edge(prev_parent, dummy)
                        Graph.add_edge(dummy, child)
                        prev_parent = dummy       
                    elif cur_parent != next_parent:
                        Graph.add_edge(prev_parent, child) 
                        prev_parent = cur_parent
                        
                else:
                    Graph.add_edge(prev_parent, child) 
            else:
                Graph.add_edge(cur_parent, child) 
                prev_parent = cur_parent
                    
        elif cur_parent == prev_parent:
            if (i+1) < num_edges:
                next_edge = dfs_edges[i+1]
                next_parent = next_edge[0]
                if cur_parent == next_parent:
                    dummy = dummy - 1
                    Graph.add_edge(cur_parent, dummy)
                    Graph.add_edge(dummy, child)
                    prev_parent = dummy            
                elif cur_parent != next_parent:
                    # do not need to add dummy node - child becomes parent's node right child
                    Graph.add_edge(cur_parent, child)
                    prev_parent = cur_parent
            else:
                Graph.add_edge(cur_parent, child)
              
    return Graph

def createBinEdgeDict(bfs_bin_edges, tree):
    bin_edges_Dict = {}
    for edge in bfs_bin_edges:
        u = edge[0]; v = edge[1]
        if u >= 0 and v < 0:
            bin_edges_Dict[v] = u
        if u < 0 and v < 0:
            parent = tree.predecessors(v)
            u = parent[0]
            while u < 0: 
                parent = tree.predecessors(u)
                # print 'Parent:',parent
                u = parent[0]
            
            bin_edges_Dict[v] = u
            
    return bin_edges_Dict

def create_LRChildren(bfs_bin_edges):
    child_dict = {}
    for edge in bfs_bin_edges:
        parent = edge[0]; child = edge[1]
        if child < 0:
            if parent not in child_dict:
                child_dict[parent] = {}
                child_dict[parent]['R'] = child
            elif parent in child_dict and child_dict[parent]['R']:
                prev_child = child_dict[parent]['R']
                child_dict[parent]['L'] = prev_child
                child_dict[parent]['R'] = child
            else:
                child_dict[parent]['R'] = child
        if child >= 0:            
            if parent not in child_dict:
                child_dict[parent] = {}
                child_dict[parent]['R'] = child
            else:
                child_dict[parent]['L'] = child
                
    return child_dict 

# creates a dictionary with the prizes of all sub-trees of a node
def createEdgeCosts(orig_edges, tree, binarized_tree):
    edge_list = binarized_tree.edges(data=False)
    edge_costs = {}
    for edge in edge_list:
        u = edge[0]; v = edge[1]
        if u < 0 and v < 0:
            edge_costs[(u,v)] = 0
        elif u >= 0 and v < 0:
            edge_costs[(u,v)] = 0
        elif u >= 0 and v >= 0:  
            if (u,v) in orig_edges:
                edge_costs[(u,v)] = orig_edges[(u,v)]
            else:
                edge_costs[(u,v)] = orig_edges[(v,u)]
        elif u < 0 and v >= 0:
            parent = tree.predecessors(v)
            if (parent[0],v) in orig_edges:
                edge_costs[(u,v)] = orig_edges[(parent[0],v)]
            else:
                edge_costs[(u,v)] = orig_edges[(v,parent[0])]
            
    return edge_costs

# creates a dictionary with the prizes of all nodes
def createNodePrizes(orig_prizes, binarized_tree):
    node_list = binarized_tree.nodes(data=False)
    for node in node_list:
        if node not in orig_prizes:
            orig_prizes[node] = 0
        else:
            continue
            
    return orig_prizes

# creates a dictionary with the prizes of all sub-trees of a node
def createSubtreePrizes(node, binarized_tree, node_prizes, subtree_prizes):
    children = binarized_tree.successors(node) 
    prize = 0
    if not children:
        prize = node_prizes[node]
        subtree_prizes[node] = prize
        return subtree_prizes, node
    else:
        for u in children:
            (subtree_prizes, u) = createSubtreePrizes(u, binarized_tree, node_prizes, subtree_prizes)
            prize = prize + (subtree_prizes[u])
            
    subtree_prizes[node] = prize + (node_prizes[node])
    return subtree_prizes, node

def initDPArrays(subtree_prizes,D,Dp,H,Hp,L,Lp,B,Bp):
    k = 0
    
    for key,value in subtree_prizes.iteritems():
        D[key] = {}; Dp[key] = {}; H[key] = {}; Hp[key] = {}; L[key] = {}; Lp[key] = {}; B[key] = {}; Bp[key] = {};
        D[key][0] = subtree_prizes[key]; Dp[key][0] = []; 
        H[key][0] = subtree_prizes[key]; Hp[key][0] = []; 
        L[key][0] = subtree_prizes[key]; Lp[key][0] = [];
        B[key][0] = subtree_prizes[key]; Bp[key][0] = [];
        
    return D,Dp,H,Hp,L,Lp,B,Bp

def minD(k,node,node_children,edge_costs,D,Dp,L,Lp,B,Bp):
    # Compute each subproblem cost separately
    min_val_NC = float('inf'); min_path_NC = []
    min_val_CL = float('inf'); min_path_CL = []
    min_val_CR = float('inf'); min_path_CR = []
    
    # node is being a path by itself - not connecting it to any child
    # only if node is a real node (not dummy)
    if 0 <= node:
        kl = 0; kr = k-1; min_path = []
        for i in range(1,k):     
            min_path = []
            # adding node as a separate path
            node_path = []; node_path.append(node)
            min_path.append(node_path)
            
            # compute best cost from left sub-tree
            left_path = []
            if 'L' in node_children[node]:
                vl = node_children[node]['L']                
                costL = L[vl][kl]
                if kl > 0:
                    tupL = ('L',vl,kl)
                    # adding left node as a separate path
                    left_path.append(tupL)
                    min_path.append(left_path)
            else:
                costL = 0
            
            right_path = []
            if 'R' in node_children[node]:
                vr = node_children[node]['R']
                if vr >= 0:
                    costR = L[vr][kr]
                    if kr > 0:
                        tupR = ('L',vr,kr)
                        right_path.append(tupR)
                        min_path.append(right_path)
                elif vr < 0:
                    costDR = D[vr][kr]
                    costBR = B[vr][kr]
                    costR = costBR
                    if kr > 0:
                        tupR = ('B',vr,kr)
                        right_path.append(tupR)
                        min_path.append(right_path)
            else:
                costR = 0
            costT = costL + costR
            if costT <= min_val_NC:
                min_val_NC = costT
                min_path_NC = min_path


            kl+=1; kr-=1
    
    # connecting node to left child
    # for any type of node
    kl = 1; kr = k-1; min_path = []; edge_costL = 0
    if 'L' in node_children[node]:
        for i in range(1,k+1):
            min_path = []
            # compute best cost from left sub-tree
            left_path = []
            if 'L' in node_children[node]:
                vl = node_children[node]['L']
                costL = D[vl][kl]; 
                edge_costL = edge_costs[(node,vl)]
                if kl > 0:
                    left_path.append(node)
                    tupL = ('D',vl,kl)
                    left_path.append(tupL)
                    min_path.append(left_path)

            else:
                costL = 0

            right_path = []
            if 'R' in node_children[node]:
                vr = node_children[node]['R']
                if vr >= 0:
                    costR = L[vr][kr]
                    if kr > 0:
                        tupR = ('L',vr,kr)
                        right_path.append(tupR)
                        min_path.append(right_path)
                elif vr < 0:
                    costDR = D[vr][kr]
                    costBR = B[vr][kr]
                    costR = costBR
                    if kr > 0:
                        tupR = ('B',vr,kr)
                        right_path.append(tupR)
                        min_path.append(right_path)
            else:
                costR = 0

            costT = costL + costR + edge_costL
            if costT <= min_val_CL:
                min_val_CL = costT
                min_path_CL = min_path

            kl+=1; kr-=1

    # connecting node to right child
    # for any type of node
    kl = 0; kr = k; min_path = []; edge_costR = 0
    if 'R' in node_children[node]:
        for i in range(0,k):
            min_path = []
            right_path = []
            if 'R' in node_children[node]:
                vr = node_children[node]['R']
                costR = D[vr][kr]; 
                edge_costR = edge_costs[(node,vr)]
                if kr > 0:
                    right_path.append(node)
                    tupR = ('D',vr,kr)
                    right_path.append(tupR)
                    min_path.append(right_path)
            else:
                costR = 0

            # compute best cost from right sub-tree
            left_path = []
            if 'L' in node_children[node]:
                vl = node_children[node]['L']
                costL = L[vl][kl];
                if kl > 0:
                    tupL = ('L',vl,kl)
                    left_path.append(tupL)
                    min_path.append(left_path)
            else:
                costL = 0
            costT = costL + costR + edge_costR

            if costT <= min_val_CR:
                min_val_CR = costT
                min_path_CR = min_path
            kl+=1; kr-=1
  
    if min_val_CR <= min_val_CL and min_val_CR <= min_val_NC:
        min_val = min_val_CR; min_path = min_path_CR
    elif min_val_CL <= min_val_CR and min_val_CL <= min_val_NC:
        min_val = min_val_CL; min_path = min_path_CL
    elif min_val_NC <= min_val_CR and min_val_NC <= min_val_CL:
        min_val = min_val_NC; min_path = min_path_NC
    else:
        print 'Should not be here 1'
    
    # print
    return (min_val,min_path)


def minH(k,node,node_children,edge_costs,D,Dp,L,Lp,H,Hp):
    # Compute each subproblem cost separately
    min_val_HB = float('inf'); min_path_HB = []
    min_val_HD = float('inf'); min_path_HD = []

    # there exists a better horizontal path below node 
    # only if node is a dummy node
    min_path = []; horizontal_path = []
    if 0 > node:
        kl = 0; kr = k; min_path = []; edge_costR = 0; horizontal_path = []
        for i in range(1,k):
            min_path = []
            # compute best cost from left sub-tree
            if 'L' in node_children[node]:
                vl = node_children[node]['L']
                costL = L[vl][kl]
                if kl > 0:
                    tupL = ('L',vl,kl)
                    horizontal_path.append(tupL)
            else:
                costL = 0

            if 'R' in node_children[node]:
                vr = node_children[node]['R']
                costR = H[vr][kr]; 
                edge_costR = edge_costs[(node,vr)]
                if kr > 0:
                    tupR = ('H',vr,kr)
                    horizontal_path.append(tupR)
            else:
                costR = 0

            costT = costL + costR + edge_costR
            horizontal_path.insert(0, node)
            min_path.append(horizontal_path)
            
            if costT <= min_val_HB:
                min_val_HB = costT
                min_path_HB = min_path


            kl+=1; kr-=1
    
    # connecting node to left child and right child to make a horizontal path
    # for any type of node
    kl = 1; kr = k; min_path = []; edge_costL = 0; edge_costR = 0; horizontal_path = []
    for i in range(1,k+1):
        if 'L' in node_children[node] and 'R' in node_children[node]:
            min_path = []; horizontal_path = []
            # compute best cost from left sub-tree
            if 'L' in node_children[node]:

                vl = node_children[node]['L']
                costL = D[vl][kl]; 
                edge_costL = edge_costs[(node,vl)]
                if kl > 0:
                    tupL = ('D',vl,kl)
                    horizontal_path.append(tupL)
            else:
                costL = 0

            if 'R' in node_children[node]:
                vr = node_children[node]['R']
                costR = D[vr][kr]; 
                edge_costR = edge_costs[(node,vr)]
                if kr > 0:
                    tupR = ('D',vr,kr)
                    horizontal_path.append(tupR)
            else:
                costR = 0

            costT = costL + costR + edge_costL + edge_costR
            horizontal_path.insert(0, node)
            min_path.append(horizontal_path)
            if costT <= min_val_HD:
                min_val_HD = costT
                min_path_HD = min_path
            
        
        kl+=1; kr-=1   
    if min_val_HB <= min_val_HD:
        min_val = min_val_HB; min_path = min_path_HB
    elif min_val_HD < min_val_HB:
        min_val = min_val_HD; min_path = min_path_HD
    else:
        print 'Should not be here 2'
    
    # print
    return (min_val,min_path)

def minB(k,node,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp):
    inf = float('inf')
    min_val_B = float('inf'); min_path_B = []
    
    # there exist better paths that do not include the node
    # node is not part of the path
    # for any type of node
    kl = 0; kr = k; min_path = [];
    left_path = []; right_path = []
    for i in range(0,k+1):       
        left_path = []; right_path = []
        min_path = []
        # compute best cost from right sub-tree
        if 'L' in node_children[node]:
            vl = node_children[node]['L']
            costL = L[vl][kl];
            if kl > 0:
                tupL = ('L',vl,kl)
                left_path.append(tupL)
                min_path.append(left_path)
        else:
            costL = 0
        
        if 'R' in node_children[node]:
            vr = node_children[node]['R']
            costR = L[vr][kr];
            if kr > 0:
                tupR = ('L',vr,kr)
                right_path.append(tupR)
                min_path.append(right_path)
        else:
            costR = 0

        costT = costL + costR + node_prizes[node]           
        
        if costT <= min_val_B and min_path:
            min_val_B = costT
            min_path_B = min_path
                    
        kl+=1; kr-=1

    min_val = min_val_B
    min_path = min_path_B
    
    return (min_val,min_path)

def minL(k,node,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp,B,Bp):
    # Compute each subproblem cost separately
    min_val_D = float('inf'); min_path_D = []
    min_val_H = float('inf'); min_path_H = []
    min_val_WN = float('inf'); min_path_WN = []

    # Existing path that includes node
    min_val_D = D[node][k]; min_path_D = Dp[node][k];
    min_val_H = H[node][k]; min_path_H = Hp[node][k];
    min_val_WN = B[node][k]; min_path_WN = Bp[node][k];
    
    if min_val_WN <= min_val_D and min_val_WN <= min_val_H:
        min_val = min_val_WN; min_path = min_path_WN
    elif min_val_D <= min_val_WN and min_val_D <= min_val_H:
        min_val = min_val_D; min_path = min_path_D
    elif min_val_H <= min_val_D and min_val_H <= min_val_WN:
        min_val = min_val_H; min_path = min_path_H
    else:
        print 'Should not be here 3'
    # print
    
    return (min_val,min_path)

def kPathDP(k,node,tree,root,node_children,node_prizes,edge_costs,subtree_prizes,D,Dp,H,Hp,L,Lp,B,Bp):
    inf = float('inf')
    children = tree.successors(node) 

    leaves = []
    if not children:
        path = []
        path.append(node)
        # D: node is a path itself H: One node cannot make horizontal path L: By definition it is going to be D

        if k == 1:
            D[node][k] = 0; Dp[node][k] = path; 
            H[node][k] = inf; Hp[node][k] = []; 
            L[node][k] = 0; Lp[node][k] = path;
            B[node][k] = inf; Bp[node][k] = [];
        else:
            D[node][k] = inf; Dp[node][k] = []; 
            H[node][k] = inf; Hp[node][k] = []; 
            L[node][k] = inf; Lp[node][k] = []; 
            B[node][k] = inf; Bp[node][k] = [];
        
        leaves.append(node)
        return D,Dp,H,Hp,L,Lp,B,Bp,node,leaves
    else:
            
        for u in children:

            (D,Dp,H,Hp,L,Lp,B,Bp,u,leaves) = kPathDP(k,u,tree,root,node_children,node_prizes,edge_costs,                                                subtree_prizes,D,Dp,H,Hp,L,Lp,B,Bp)
            
            if k not in D[u]:
                (min_valueD,min_pathD) = minD(k,u,node_children,edge_costs,D,Dp,L,Lp,B,Bp)
                D[u][k] = min_valueD; Dp[u][k] = min_pathD
                (min_valueH,min_pathH) = minH(k,u,node_children,edge_costs,D,Dp,L,Lp,H,Hp)
                H[u][k] = min_valueH; Hp[u][k] = min_pathH
                (min_valueB,min_pathB) = minB(k,u,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp) 
                B[u][k] = min_valueB; Bp[u][k] = min_pathB
                (min_valueL,min_pathL) = minL(k,u,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp,B,Bp) 
                L[u][k] = min_valueL; Lp[u][k] = min_pathL

                
    if node == root:
        (min_valueD,min_pathD) = minD(k,node,node_children,edge_costs,D,Dp,L,Lp,B,Bp)
        D[node][k] = min_valueD; Dp[node][k] = min_pathD
        (min_valueH,min_pathH) = minH(k,node,node_children,edge_costs,D,Dp,L,Lp,H,Hp)
        H[node][k] = min_valueH; Hp[node][k] = min_pathH
        (min_valueB,min_pathB) = minB(k,node,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp) 
        B[node][k] = min_valueB; Bp[node][k] = min_pathB
        (min_valueL,min_pathL) = minL(k,node,node_children,edge_costs,node_prizes,D,Dp,L,Lp,H,Hp,B,Bp) 
        L[node][k] = min_valueL; Lp[node][k] = min_pathL
        
    return D,Dp,H,Hp,L,Lp,B,Bp,node,leaves

def retrievePath(my_list,flag,flag2,bridges,k,node,T,binarized_tree,paths,cur_path,Dp,Hp,Lp,Bp):
    if T == 'L':
        moves = Lp[node][k]
    elif T == 'H':
        moves = Hp[node][k]
    elif T == 'D':
        moves = Dp[node][k]
    elif T == 'B':
        moves = Bp[node][k]
  
    # case of only one node left that is an integer
    if len(moves) == 1 and isinstance(moves[0],int):
        node = moves[0]
            
        # If new path is created then you need to close the previous path and begin a new one
        if T == 'L' and cur_path:               
            if flag == False:
                bridges.append(None)
            # closing previous path
            flag == False
            flag2 == False
            paths.append(cur_path)
            # beginning new path
            cur_path = []

        # if non-root node
        if binarized_tree.predecessors(node):
            parent_list = binarized_tree.predecessors(node)
            parent = parent_list[0]
            actual_parent = None
            first_parent = 'Root'    
            
            if cur_path and cur_path[-1] < 0:
                dummy_node = cur_path[-1]
                parent_list = binarized_tree.predecessors(dummy_node)
                parent = parent_list[0]
                actual_parent = dummy_node
                
            if cur_path and binarized_tree.predecessors(cur_path[0]):
                first_parent_list = binarized_tree.predecessors(cur_path[0])
                first_parent = first_parent_list[0]
            elif not cur_path and binarized_tree.predecessors(node):
                help_list = binarized_tree.predecessors(node)
                help_parent = help_list[0]
                if help_parent in bridges:
                    first_parent = help_parent
            else:
                first_parent = 'Root'
            if (cur_path and cur_path[-1] == parent and first_parent not in bridges):
                cur_path.append(node);
                if flag2 == False:
                    paths.append(cur_path);
                cur_path = []
                flag == False
                flag2 = False
            elif (not cur_path and first_parent not in bridges):
                cur_path.append(node);
                if flag2 == False:
                    paths.append(cur_path);
                cur_path = []
                flag == False
                flag2 = False           
            
            elif cur_path and cur_path[-1] != parent and cur_path[-1]!= actual_parent and parent in bridges:
                if flag2 == False:
                    paths.append(cur_path)               
                flag == False
                cur_path = []
                index = my_list[parent]
                if paths:
                    cur_path = paths[index]
                    cur_path.reverse()
                    cur_path.append(node)
                    del paths[index]
                    paths.insert(index, cur_path)
                    flag2 = True
                elif not paths:
                    cur_path.append(node)
                    if flag2 == False:
                        paths.append(cur_path)
            
            elif len(cur_path) == 1 and cur_path[0] < 0 and parent in bridges:
                index = my_list[parent]
                cur_path = paths[index]
                cur_path.reverse()
                cur_path.append(node)
                flag2 = True    
            elif cur_path and cur_path[0] < 0 and parent in bridges:
                if flag2 == False:
                    paths.append(cur_path)  
                                        
                flag == False
                # beginning new path
                cur_path = []
                # find which list you belong to
                index = my_list[parent]
                cur_path = paths[index]
                cur_path.reverse()
                cur_path.append(node)
                del paths[index]
                paths.insert(index, cur_path)
                flag2 = True
            elif cur_path and first_parent in (my_list and bridges):
                flag == False
                cur_path.append(node)
                if flag2 == False:
                    new_cur_path = list(cur_path)
                    index = my_list[first_parent]
                    cur_path = paths[index]
                    cur_path.reverse()
                    conc_paths = cur_path + new_cur_path
                    del paths[index] 
                    paths.insert(index, conc_paths)
                                     
                cur_path = []
                flag2 = False
                
            elif cur_path and cur_path[-1] != parent and parent not in my_list:
                cur_path.append(node);
                if flag2 == False:
                    paths.append(cur_path);
                cur_path = []
                flag == False
                flag2 = False
            elif not cur_path and first_parent in my_list and first_parent in bridges:                    
                flag == False
                index = my_list[first_parent]
                cur_path = paths[index]
                if cur_path[-1] == first_parent:
                    cur_path.append(node)
                elif cur_path[0] == first_parent:
                    cur_path.reverse()
                    cur_path.append(node)

                del paths[index]
                paths.insert(index, cur_path)
                cur_path = []
                flag2 = False
                
            elif cur_path and cur_path[-1] < 0 and parent not in bridges:                   
                cur_path.append(node);
                if flag2 == False:
                    paths.append(cur_path);
                cur_path = []
                flag == False
                flag2 = False
        else:
            print 'I am the root!'
        
        return my_list,flag,flag2,bridges,k,node,T,paths,cur_path,Dp,Hp,Lp,Bp
    elif not moves:
        flag2 = False
        return my_list,flag,flag2,bridges,k,node,T,paths,cur_path,Dp,Hp,Lp,Bp
    else: 
        moves = [item for sublist in moves for item in sublist]
        # Adding a bridging element
        if len(moves) == 3 and isinstance(moves[0],int) and moves[1][0] == 'D' and moves[2][0] == 'D':
            bridges.append(moves[0])
                
            flag = True
            flag2 = False

        node = moves[0] 
        if T == 'L' and cur_path and not isinstance(node,int):
            if flag == False:
                bridges.append(None)
            if flag2 == False:
                paths.append(cur_path)
                
                
            flag = False
            # beginning new path
            cur_path = [] 
            flag2 = False
        elif T == 'L' and cur_path and isinstance(node,int):
            if flag == False:
                bridges.append(None)
            if flag2 == False:
                paths.append(cur_path)
                
                
            flag = False
            # beginning new path
            cur_path = [] 
            flag2 = False
        
        # if node is an integer needs to be added in the path - and moves start from index 1
        if isinstance(node,int):
            idx = 1;
            if not cur_path:
                my_list[node] = len(paths)
 
            if binarized_tree.predecessors(node):
                parent_list = binarized_tree.predecessors(node)
                parent = parent_list[0]
                    
                if (cur_path and cur_path[-1] == parent) or not cur_path:         
                    cur_path.append(node)
  
                elif cur_path and cur_path[-1] != parent and parent in my_list\
                    or cur_path[0] :
                    if flag2 == False:
                        paths.append(cur_path)
                        
                    flag == False
                    cur_path = []
                    index = my_list[parent]
                    cur_path = paths[index]
                    cur_path.reverse()

                    cur_path.append(node)
                    del paths[index]
                    paths.insert(index, cur_path)
                    flag2 = True  
                else:          
                    cur_path.append(node)
                
            else:
                    
                cur_path.append(node)
            
        elif isinstance(node,list) and len(node) == 1 and isinstance(node[0],int):            
            if flag2 == False:
                paths.append(cur_path);
            if flag == False:
                bridges.append(None)
            cur_path = []
            my_list[node] = len(paths)
                
            cur_path.append(node);
            paths.append(cur_path);
            flag == False
            flag2 = False
            
        else:
            idx = 0
     
        for move in moves[idx:]:
            T = move[0]; node = move[1]; k = move[2]
            (my_list, flag,flag2,bridges,k,node,T,paths,cur_path,Dp,Hp,Lp,Bp) = \
                                    retrievePath(my_list,flag,flag2,bridges,k,node,T,binarized_tree,\
                                                                       paths,cur_path,Dp,Hp,Lp,Bp) 
            
    return my_list,flag,flag2,bridges,k,node,T,paths,cur_path,Dp,Hp,Lp,Bp

def createPathRoute(paths,bridges,bin_edges_Dict):
    from itertools import groupby
    pos_dict = {}; remove_idx = []; remove_idxs = []
    intermediate_paths = []; final_paths = []
    for path in paths:
        remove_idx = []
        for i,node in enumerate(path):
            pos_dict[node] = i
            if node < 0:
                real_parent = bin_edges_Dict[node]
                path[i] = real_parent

        remove_idxs.append(remove_idx)

    for i,path in enumerate(paths):
        path = [x[0] for x in groupby(path)]
        final_paths.append(path)
 
    return final_paths

def kBinarizedPCSP(LGraph, MST, root, ks):
    D = {}; Dp = {}; L = {}; Lp = {}; H = {}; Hp = {}; B = {}; Bp = {};
    objective_value = float('inf')
    subtree_prizes = {}
    paths = []; cur_path = []; final_paths = []
    # Functions
    node_list = MST.nodes()
    
    if len(node_list) <= ks:
        objective_value = float('inf')
        pathi = []
        for node in node_list:
            pathi.append(node)
            final_paths.append(pathi)
            pathi = []
        return (objective_value,final_paths,D,Dp,H,Hp,L,Lp,B,Bp)

    Graph = nx.Graph(MST)
    sub_graphs = nx.connected_component_subgraphs(Graph)

    big_elem = -float('inf'); max_numnodes = -float('inf') 
    
    for i, sub_graph in enumerate(sub_graphs): 
        nodes = sub_graph.nodes(data=False)
        root_comp = nodes[0]
        num_nodes = len(nodes)
        
        if num_nodes > max_numnodes:
            max_numnodes = num_nodes
            temp_tree = nx.DiGraph(sub_graph.to_directed())
            root = root_comp
    
    if max_numnodes <= ks:
        objective_value = float('inf')
        pathi = []
        for node in node_list:
            pathi.append(node)
            final_paths.append(pathi)
            pathi = []
        return (objective_value,final_paths,D,Dp,H,Hp,L,Lp,B,Bp)
    
    tree = nx.DiGraph(temp_tree)

    nodes = tree.nodes(data=False)
    root = nodes[0]

    
    LamdaGraph = nx.Graph(LGraph)
    
    orig_prizes, orig_edges = prepareNodeEdgeDict(tree,LamdaGraph)

    bfs_edges_gen = create_edges_BFS(tree,root)

    # BFS edges to construct the binarized tree
    bfs_edges = list(bfs_edges_gen)

    # Graph that is a right-binarized tree
      
    binarized_tree = binarizeTree(bfs_edges) 
    bfs_bin_edges_gen = create_edges_BFS(binarized_tree,root)

    bfs_bin_edges = list(bfs_bin_edges_gen)
    bin_edges_Dict = createBinEdgeDict(bfs_bin_edges,binarized_tree)

    node_children = create_LRChildren(bfs_bin_edges)

    edge_costs = createEdgeCosts(orig_edges, tree, binarized_tree)
    # Node Prizes dictionary of the right-binarized tree
    node_prizes = createNodePrizes(orig_prizes, binarized_tree)
    subtree_prizes, node = createSubtreePrizes(root, binarized_tree, node_prizes, subtree_prizes)

    # Initializing k=0 in DP arrays
    D,Dp,H,Hp,L,Lp,B,Bp = initDPArrays(subtree_prizes,D,Dp,H,Hp,L,Lp,B,Bp)   
    
    # Running k Paths Dynamic Program
    for k in range(1,ks+1):
        D,Dp,H,Hp,L,Lp,B,Bp,node,leaves = kPathDP(k,root,binarized_tree,root,node_children,node_prizes,                                             edge_costs,subtree_prizes,D,Dp,H,Hp,L,Lp,B,Bp)
    
    bridges = []; bridge = None; my_list = {}
    (my_list,flag,flag2,bridges,k,node,T,paths,cur_path,Dp,Hp,Lp,Bp) = \
                        retrievePath(my_list,False,False,bridges,ks,root,'L',binarized_tree,paths,cur_path,Dp,Hp,Lp,Bp)
    if flag == False:
        bridges.append(None)

    final_paths = createPathRoute(paths,bridges,bin_edges_Dict)
    objective_value = L[root][ks]

    return (objective_value,final_paths,D,Dp,H,Hp,L,Lp,B,Bp)

def runK_DPPC(LamdaGraph,OGraph,k):
    Graph = nx.Graph(OGraph)
    LGraph = nx.Graph(LamdaGraph)
    
    node_list = Graph.nodes()
    root = node_list[0]
    (objective_value,final_paths,D,Dp,H,Hp,L,Lp,B,Bp) = kBinarizedPCSP(LGraph, Graph, root, k)   
    final_edges = createEdgesFromK_DPPath(final_paths)
    return (final_paths, final_edges,objective_value)

def runKPCSP(TreeGraph,LGraph,k):
    MST = nx.Graph(TreeGraph)
    GraphLamda = nx.Graph(LGraph)                        
    (min_path_DP, min_edges_DP, min_obj_value_DP) = \
                          runK_DPPC(MST,GraphLamda,k)
        
    return (min_path_DP,min_edges_DP,min_obj_value_DP)

def runGreedyDP(GreedyMatrix,GreedyMatrixNodes,GreedyMatrixEdges,tot_prize,k):
    print
    min_score = tot_prize
    OPT = {}; OPT_Path = {}
     
    for i in range(0,k+1):
        OPT[i] = {}; OPT_Path[i] = {}
        min_score = tot_prize
        if i == 0:   
            for l in range(0,k+1):
                OPT[i][l] = tot_prize
                OPT_Path[i][l] = []
        else:
            for j in range(0,k+1): 
                solution = ()
                for l in range(0,j+1):
                    if l == j:
                        score = OPT[i-1][l] + GreedyMatrix[i][j-l]
                    else:
                        score = OPT[i-1][l] + GreedyMatrix[i][j-l] - tot_prize
                    if score < min_score:
                        min_score = score
                        solution = ([i-1,l],[i,j-l])
 
                OPT[i][j] = min_score
                OPT_Path[i][j] = solution  
    
    
    # print 'OPT MATRIX:'
    # for key, value in OPT.iteritems():
    #     print 'key:',key,'value:',value
    # print   
    # print 'OPT PATH:'
    # for key, value in OPT_Path.iteritems():
    #     print 'key:',key,'value:',value
    # print
    
    # print 'Greedh Matrix Nodes:',GreedyMatrixNodes
    # print
    idx = k
    nxt_paths = k
    DPSolutionPath = []
    DPSolutionEdges = []
    while idx != 0 and OPT_Path[idx][nxt_paths]:
        (nxt,curr) = OPT_Path[idx][nxt_paths]
        curr_paths = curr[1]
        paths = GreedyMatrixNodes[idx][curr_paths]
        edges = GreedyMatrixEdges[idx][curr_paths]
        for path in paths:
            DPSolutionPath.append(path)
        for edge in edges:
            DPSolutionEdges.append(edge)
             
        idx = nxt[0]
        nxt_paths = nxt[1]

    opt_score = OPT[k][k]
    return DPSolutionPath,DPSolutionEdges,opt_score

def initComponents(OriginalGraph,GraphLamda,k):
    subgraphs = []
    Graph = nx.Graph(OriginalGraph)
    # create prize-collecting steiner tree from the graph
    (index_dictionary, node_dictionary, edges_dictionary, edges, prizes, costs) = prepareFastPCST(GraphLamda)
    vertices, edges = pc.pcst_fast(edges, prizes, costs, -1, k, 'strong', 0)
    (dec_nodes, decoded_edges) = decodeFast(index_dictionary,edges_dictionary,vertices,edges)
     
    decoded_nodes = []
    decoded_nodes.append(dec_nodes) 
     
    # create induced graph based on output from indyk
    if len(decoded_edges) > 0:
        Graph = createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)
     
    sub_graphs = nx.connected_component_subgraphs(Graph)
    points = []
    # subgraphs contains pairs of nodes and edges that correspond to a graph
    Graph2 = nx.Graph(Graph)
    for i, sub_graph in enumerate(sub_graphs):
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)
        Graph2 = nx.Graph(Graph)
        points.append(nodes)
        if len(edges) > 0:
            Graph2 = createInducedSubgraph(edges,nodes)
        else:
            
            node_list = Graph2.nodes()
            remove = list(set(node_list) - set(nodes))
            Graph2.remove_nodes_from(remove) 
             
        subgraphs.append(Graph2)

    return subgraphs

def runkPaths(OriginalGraph,LamdaOriginalGraph,k,lamda):
    inf = float("inf")
    orig_node_values = OriginalGraph.nodes(data=True)
    node_values = LamdaOriginalGraph.nodes(data=True)

    DPSolutionPath = []; DPSolutionEdges = []
    DPSolutionObjVal = []
    solution_paths = []; solution_edges = []; solution_obj_values = []
    min_paths = []; min_edges = []
    GreedyMatrix = {}; GreedyMatrixNodes = {};  GreedyMatrixEdges = {}
    counter = 0
    node_prizes = getPrizes(LamdaOriginalGraph)
    tot_prize = computeObjectiveValueGraph(LamdaOriginalGraph)
    Graph = nx.Graph(OriginalGraph)
    LamdaGraph = nx.Graph(LamdaOriginalGraph)
    subgraphs = initComponents(Graph,LamdaGraph,k)
    MSTS = []
    to_plot = []
    
    # for each subgraph   
    for i,MST in enumerate(subgraphs):
        Graph = nx.Graph(OriginalGraph)
        LamdaGraph = nx.Graph(LamdaOriginalGraph)      
        PCGraph = nx.Graph(MST)
        
        GreedyMatrix[i+1] = {}
        GreedyMatrixNodes[i+1] = {}
        GreedyMatrixEdges[i+1] = {}
        nodes = MST.nodes(data=False)
        edges = MST.edges(data=False)
        MSTS.append(edges)
        to_plot.append(nodes)
        objective_value = tot_prize 
        prev_objective = objective_value
        solution_obj_values.append(0)
        GreedyMatrix[i+1][0] = 0
        GreedyMatrixNodes[i+1][0] = []
        GreedyMatrixEdges[i+1][0] = []

        while (counter < k):
            TreeGraph = nx.Graph(PCGraph)
            LamdaGraph = nx.Graph(LamdaOriginalGraph)
            if (counter > nx.number_of_nodes(MST)):
                solution_obj_values.append(prev_objective)
                GreedyMatrix[i+1][counter+1] = inf # prev_objective
                GreedyMatrixNodes[i+1][counter+1] = []
                GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                GreedyMatrixEdges[i+1][counter+1] = []
                GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
            else:   
                tree_nodes = TreeGraph.nodes(data=False)
                num_tree_nodes = len(tree_nodes)
                if num_tree_nodes == 1:
                    if counter+1 == 1:
                        nd = tree_nodes[0]
                        prize = LamdaOriginalGraph.node[nd]['prize']
                        penalty = tot_prize - prize
                        min_paths.append([nd])
                        GreedyMatrix[i+1][counter+1] = penalty
                        solution_obj_values.append(penalty)
                        GreedyMatrixNodes[i+1][counter+1] = []
                        GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                        GreedyMatrixEdges[i+1][counter+1] = []
                        GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
                        prev_objective = penalty
                    else:
                        solution_obj_values.append(prev_objective)
                        GreedyMatrix[i+1][counter+1] = inf # prev_objective
                        GreedyMatrixNodes[i+1][counter+1] = []
                        GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                        GreedyMatrixEdges[i+1][counter+1] = []
                        GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
                    
                else:   
                    nodes_lamda = LamdaGraph.nodes(data=True)
                    (min_paths,min_edges,min_obj_value) = runKPCSP(LamdaGraph,TreeGraph,counter+1)
                                       
                    if not min_edges:
                        nd = min_paths[0]
                        prize = LamdaOriginalGraph.node[nd[0]]['prize']
                        penalty = tot_prize - prize
                        GreedyMatrix[i+1][counter+1] = penalty
                        solution_obj_values.append(penalty)
                        GreedyMatrixNodes[i+1][counter+1] = []
                        GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                        GreedyMatrixEdges[i+1][counter+1] = []
                        GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
                        prev_objective = penalty
                    else:
                        objective_value,penalty,cost = \
                            computeObjectiveValue(LamdaGraph,min_paths,min_edges,False)

                        solution_obj_values.append(objective_value)
                        GreedyMatrix[i+1][counter+1] = objective_value
                        GreedyMatrixNodes[i+1][counter+1] = []
                        GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                        GreedyMatrixEdges[i+1][counter+1] = []
                        GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
                        prev_objective = objective_value
     
                 
            counter+=1
             
        DPSolutionPath.append(min_paths); DPSolutionEdges.append(min_edges)
        DPSolutionObjVal.append(solution_obj_values)
        solution_paths = []; solution_edges = []; min_paths = []; min_edges = []
        solution_obj_values = [];
        counter = 0

    DPSolutionPath,DPSolutionEdges,opt_score\
            = runGreedyDP(GreedyMatrix,GreedyMatrixNodes,GreedyMatrixEdges,tot_prize,k)
        
    return DPSolutionPath,DPSolutionEdges
        
def createInterDictionary(filename):
    intersection_coord = {} 
    reader = csv.reader(open(filename, 'r'))
    counter = 0
    for row in reader:
        key, value = row     
        intersection_coord[key] = value     
        
    return intersection_coord

    
def createInducedSubgraph(edges,nodes):
    Graph = nx.Graph()
    for edge in edges:
        u = edge[0]; v = edge[1]
        Graph.add_edge(u,v)

    for node in nodes:
        if not Graph.has_node(node):
            Graph.add_node(node)
            
    return Graph
       
def applyLamda(GraphInit,lamda):  
    Graph = nx.Graph(GraphInit)
    for node in Graph.nodes(data=False):
        Graph.node[node]['prize'] = Graph.node[node]['prize']*(lamda)

    return Graph

def totalNode(Graph):
    sum_prize = 0
    for node in Graph.nodes(data=False):
        sum_prize+=Graph.node[node]['prize']
        
    return sum_prize

def totalEdge(Graph):
    sum_edge = 0
    for u,v in Graph.edges(data=False):
        sum_edge+=Graph.edge[u][v]['weight']
        
    return sum_edge

def computeTotalPrize(OriginalGraph,penalties):
    prize = 0
    for node in penalties:
        prize+=OriginalGraph.node[node]['prize']  
    return prize


def computeTotalCost(OriginalGraph,kPathsEdgeComponent):
    cost = 0
    counter = 0

    for edges in kPathsEdgeComponent: 
        for edge in edges:
            u = edge[0]
            v = edge[1]
            counter+=1
            
            if OriginalGraph.has_edge(u,v):
                cost+=OriginalGraph.edge[u][v]['weight']
            elif OriginalGraph.has_edge(v, u):
                cost+=OriginalGraph.edge[v][u]['weight']
            else:
                print 'Error'

    return cost
                                                                                                                        
def computeObjectiveValue(OriginalGraph,kPathsSolutionComponent,kPathsEdgeComponent,print_output):
    tot_nodes = OriginalGraph.nodes(data=False)

    length_input = 0
    component_nodes = []

    for component in kPathsSolutionComponent:
        component_nodes = component_nodes + component
        length_input+=len(component)
    
    penalty_nodes = list(set(tot_nodes)^set(component_nodes))
    
    obj_prize = computeTotalPrize(OriginalGraph,penalty_nodes)

    obj_cost = computeTotalCost(OriginalGraph,kPathsEdgeComponent)

    objective_value = obj_prize + obj_cost
    if print_output:
        print 'Final penalty is:',obj_prize
        print 'Final cost is:',obj_cost
    return objective_value,obj_prize,obj_cost
                 
def findLargestConnectedComponent(OriginalGraph):
    total_nodes = 0
    max_num_nodes = -float('inf')
    Graph = nx.Graph(OriginalGraph)
    node_prizes = getPrizes(Graph)
    sub_graphs = nx.connected_component_subgraphs(Graph)
    to_plot = []
    for i, sub_graph in enumerate(sub_graphs):
        node_list = sub_graph.nodes(data=False)
        num_nodes = len(sub_graph.nodes(data=False))
        total_nodes += num_nodes
        if num_nodes >= max_num_nodes:
            LargestComponent = nx.Graph(sub_graph)
            max_num_nodes = num_nodes
    
    return LargestComponent

    
def main():
    # graph in .gml format
    graph_file = str(sys.argv[1])
    # number of output trees
    k = int(sys.argv[2])
    # normalization factor on node prizes
    lamda = float(sys.argv[3])
    
    # uncomment to plot road network solutions
    # intersections_file = str(sys.argv[4])
    # intersection_coordinates = createInterDictionary(intersections_file)
    
    
    FileGraph = nx.read_gml(graph_file)
    
    LargestComponent = findLargestConnectedComponent(FileGraph)
    OriginalGraph = nx.Graph(LargestComponent)
    
    # applying normalization coefficient to node prizes
    LamdaGraph = applyLamda(OriginalGraph,lamda)
    Graph = nx.Graph(LamdaGraph)
    
    num_nodes = nx.number_of_nodes(OriginalGraph)
    num_edges = nx.number_of_edges(OriginalGraph)
    
    print 'k:',k
    print 'lambda:',lamda
    print 'Number of Nodes: ',num_nodes        
    print 'Number of Edges: ',num_edges  
    print
    
    if k > num_nodes:
        print 'Number of nodes should be greater or equal to the number of clusters.'
    else:     
        print 'Finding',k,'tours'
        print 'Running k Paths...'
        (kPathsSolutionComponent,kPathsEdgeComponent)\
            = runkPaths(OriginalGraph,Graph,k,lamda)

        objective_value_kPaths,penalty,cost=\
            computeObjectiveValue(LamdaGraph,kPathsSolutionComponent,kPathsEdgeComponent,False)

        print 'Solution cost is:',objective_value_kPaths       
        
if __name__ == '__main__':
    main()