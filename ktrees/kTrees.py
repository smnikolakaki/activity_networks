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

def runkTrees(OriginalGraph,k,pruning):
    kTreesSolutionComponent = []; kTreesEdgeComponent = []
    
    Graph = nx.Graph(OriginalGraph)
    node_list = Graph.nodes(data=False) 
    
    (index_dictionary, node_dictionary, edges_dictionary, edges, prizes, costs) = prepareFastPCST(Graph)

    vertices, edges = pc.pcst_fast(edges, prizes, costs, -1, k, pruning, 0)

    (dec_nodes, decoded_edges) = decodeFast(index_dictionary,edges_dictionary,vertices,edges)
 
    decoded_nodes = []
    decoded_nodes.append(dec_nodes) 
    if len(decoded_edges) > 0:
        Graph = createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)

    sub_graphs = nx.connected_component_subgraphs(Graph)
    
    for i, sub_graph in enumerate(sub_graphs):    
        kTreesSolutionComponent.append(sub_graph.nodes())
        kTreesEdgeComponent.append(sub_graph.edges())
    
    return kTreesSolutionComponent,kTreesEdgeComponent
        
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
    
    # print '2 type of graph:',type(Graph)
    for node in nodes:
        if not Graph.has_node(node):
            # print 'node:',node
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
    print
    for node in penalties:
        # print 'node:',node
        prize+=OriginalGraph.node[node]['prize']
    # print    
    return prize


def computeTotalCost(OriginalGraph,kTreesEdgeComponent):
    cost = 0
    counter = 0
    # print OriginalGraph.edges(data=False)
    # print
    for edges in kTreesEdgeComponent: 
        # print 'Edges:',edges
        for edge in edges:
            # print 'Edge:',edge
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
                                                                                                                        
def computeObjectiveValue(OriginalGraph,kTreesSolutionComponent,kTreesEdgeComponent,print_output):
    tot_nodes = OriginalGraph.nodes(data=False)

    length_input = 0
    component_nodes = []

    for component in kTreesSolutionComponent:
        # print 'Component:',component
        component_nodes = component_nodes + component
        length_input+=len(component)
    
    penalty_nodes = list(set(tot_nodes)^set(component_nodes))
    
    obj_prize = computeTotalPrize(OriginalGraph,penalty_nodes)

    obj_cost = computeTotalCost(OriginalGraph,kTreesEdgeComponent)

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
    
    # applying normalization coefficient to node prizes -- TODO OrigGraph == LamdaGraph
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
        print 'Finding',k,'trees'
        print 'Running k Trees...'
        (kTreesSolutionComponent,kTreesEdgeComponent)\
            = runkTrees(Graph,k,"strong")

        print 'K Trees Solution Component:',kTreesSolutionComponent
        print 'K Trees Edges Component:',kTreesEdgeComponent
        objective_value_kTrees,penalty,cost=\
            computeObjectiveValue(LamdaGraph,kTreesSolutionComponent,kTreesEdgeComponent,False)

        print 'Solution cost is:',objective_value_kTrees        
        
if __name__ == '__main__':
    main()