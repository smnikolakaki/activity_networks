from __future__ import division

import PCST_Greedy as pcstg
import pcst_fast as pc
import RoadNetwork as rn
import PC_Path as pcp
import PC_Tour as pcpa
import ExperimentsLevelsGreedySyntheticUniform as expu
import DensityClustering as dc
import PC_TreePath as ptp
import BinarizedTree as bt
import geojson
from geojson import LineString, Feature, FeatureCollection

import pandas as pd
import numpy as np
import metrics as metr
import time
import collections
import math 
import random as rd
import numpy as np
import networkx as nx
import sys
import folium
import webbrowser
import os
import fnmatch
import glob
import csv
from selenium import webdriver 
from PIL import Image

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def printEdges(Graph):
    #print 'I am here'
    edge_attrs = nx.get_edge_attributes(Graph, "weight")
    for edge,weight in edge_attrs.items():
        # active  = Graph.edge[edge[0]][edge[1]]['active']
        num = Graph.edge[edge[0]][edge[1]]['num']
        print "Edge: ",edge," with cost: ",weight, " with number of active components: ",num# ," and is active: ",active

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

def computeTreePenalties(DBSCANSolutionNodes,OriginalGraph):
    tot_nodes = OriginalGraph.nodes(data=False)

    length_input = 0
    component_nodes = []

    for component in DBSCANSolutionNodes:
        # print 'Component:',component
        component_nodes = component_nodes + component
    
    penalties = list(set(tot_nodes)^set(component_nodes))

    obj_prize = computeTotalPrize(OriginalGraph,penalties)  
    
    return obj_prize
    
def computeTreeEdges(DBSCANSolutionEdges,OriginalGraph):
    component_edges = []
    cost = 0
    for component in DBSCANSolutionEdges:
        component_edges = component_edges + component
            
    for edge in component_edges:
        u = edge[0]; v = edge[1]
        cost+=OriginalGraph.edge[u][v]['weight']
    
    return cost

def plotMap(points, file_out, file_img, intersection_coordinates, node_prizes):
    prizes = []
    
    # scaling parameters for visual effects
    for key,value in node_prizes.items():
        prizes.append(value)
    # print 'MV',max_value,'MV',min_value
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
    # Boston
    # mean_lat = 42.369809; mean_lon = -71.096267
    
    # SF
    # mean_lat = 37.76055; mean_lon = -122.435712
    
    map_osm = folium.Map(location=[mean_lat, mean_lon], zoom_start=14,\
                        tiles = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                         # tiles = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png")
                         # tiles = "https://api.mapbox.com/styles/v1/smnikolakaki/cj11403nz001w2sqs32tsuj8u/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1Ijoic21uaWtvbGFrYWtpIiwiYSI6ImNpejJybjZuMDA0dDQycXA4MGtrMHlrM2UifQ.N8W99VYSvSPl_-aS5jCGwA",
            attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy;<a href="https://carto.com/attribution">CARTO</a>')

    # add a marker for every record in the filtered data, use a clustered view
    for component, nds in enumerate(points[0:limit]):
        nodes = []
        if isinstance(nds,int):
            nodes.append(nds)
        else:
            nodes = nds
            
        for node in nodes:
            prize = node_prizes[node]
            coord = intersection_coordinates[str(node)]
            lat_long = [x.strip() for x in coord.split(',')]
            lat = float(lat_long[0])
            lon = float(lat_long[1])
            popup = 'component: %d\t node: %s\t prize: %s' % (component, str(node),str(prize))
            node_color = colors[component % len(colors)]
            map_osm.circle_marker(location=[lat, lon], radius=5,
                                  line_color=node_color,
                                  popup=popup, fill_opacity=40,
                                  fill_color=node_color)

    # map_osm.lat_lng_popover()
    map_osm.create_map(path=file_out)
    browser = webdriver.Firefox()
    browser.set_window_size(1680, 1050)
    browser.maximize_window()
    # browser.manage().window().maximize();
    browser.get('file:///'+file_out)
    browser.execute_script("document.body.style.zoom='%'") 
    time.sleep(5) # delays for 5 seconds
    browser.save_screenshot(file_img)
    browser.quit()
    # # Load the original image:
    img = Image.open(file_img)
    width, height = img.size
    print 'Width:',width,'Height:',height
    # Boston
    img2 = img.crop((600, 300, width-600, height-350))
    # SanFrancisco
    # # img2 = img.crop((300, 300, width-300, height-240))
    # # img2.save(file_img)
    img2.save(file_img)
    
def plotPath(points, file_out, file_img, intersection_coordinates, node_prizes):
    prizes = []
    # print 'Points:',points
    # scaling parameters for visual effects
    for key,value in node_prizes.items():
        prizes.append(value)
    # print 'MV',max_value,'MV',min_value
    colors = ["#c30bd4", "#2096ba", "#5870ff", "#a70100", "#5da4ff",
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
    # Boston
    # mean_lat = 42.369809; mean_lon = -71.096267
    
    # SF
    # mean_lat = 37.76055; mean_lon = -122.435712
    
    map_osm = folium.Map(location=[mean_lat, mean_lon], zoom_start=12,\
                         tiles = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                         # tiles = "https://api.mapbox.com/styles/v1/smnikolakaki/cj11403nz001w2sqs32tsuj8u/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1Ijoic21uaWtvbGFrYWtpIiwiYSI6ImNpejJybjZuMDA0dDQycXA4MGtrMHlrM2UifQ.N8W99VYSvSPl_-aS5jCGwA",
            attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy;<a href="https://carto.com/attribution">CARTO</a>')

    # add a marker for every record in the filtered data, use a clustered view
    for component, nds in enumerate(points[0:limit]):
        nodes = []
        if isinstance(nds,int):
            nodes.append(nds)
        else:
            nodes = nds
        coordinates = [] 
        if len(nodes) == 1:
            for node in nodes:
                prize = node_prizes[node]
                coord = intersection_coordinates[str(node)]
                lat_long = [x.strip() for x in coord.split(',')]
                lat = float(lat_long[0])
                lon = float(lat_long[1])
                popup = 'component: %d\t node: %s\t prize: %s' % (component, str(node),str(prize))
                # node_color = colors[component % len(colors)]
                # map_osm.circle_marker(location=[lat, lon], radius=1,
                #                       line_color=node_color,
                #                       popup=popup, fill_opacity=1,
                #                       fill_color=node_color)
                location = [lat,lon]
                location2 = [lat+0.0005,lon+0.0005]
                coordinates.append(location)
                coordinates.append(location2)
        else:
            for node in nodes:
                prize = node_prizes[node]
                coord = intersection_coordinates[str(node)]
                lat_long = [x.strip() for x in coord.split(',')]
                lat = float(lat_long[0])
                lon = float(lat_long[1])
                popup = 'component: %d\t node: %s\t prize: %s' % (component, str(node),str(prize))
                # node_color = colors[component % len(colors)]
                # map_osm.circle_marker(location=[lat, lon], radius=1,
                #                       line_color=node_color,
                #                       popup=popup, fill_opacity=1,
                #                       fill_color=node_color)
                location = [lat,lon]
                coordinates.append(location)
            
        node_color = colors[component % len(colors)]
        my_PolyLine=folium.PolyLine(locations=coordinates,weight=3,color=node_color, opacity=1)
        map_osm.add_children(my_PolyLine)

    # map_osm.lat_lng_popover()
    map_osm.create_map(path=file_out)
    browser = webdriver.Firefox()
    browser.set_window_size(1680, 1050)
    browser.maximize_window()
    # browser.manage().window().maximize();
    browser.get('file:///'+file_out)
    browser.execute_script("document.body.style.zoom='%'") 
    time.sleep(5) # delays for 5 seconds
    browser.save_screenshot(file_img)
    browser.quit()
    # # Load the original image:
    img = Image.open(file_img)
    width, height = img.size
    print 'Width:',width,'Height:',height
    # Boston
    img2 = img.crop((400, 300, width-400, height-350))
    # SanFrancisco
    # # img2 = img.crop((300, 300, width-300, height-240))
    # # img2.save(file_img)
    img2.save(file_img)

    # webbrowser.open('file://' + os.path.realpath(file_out))


def runIndyk(OriginalGraph,lamda,k,directory_plots,city,intersection_file,pruning):
    # print
    IndykObjValComponent = []
    IndykSolutionComponent = []; IndykEdgeComponent = []
    DPSolutionPath = []; DPEdgePath = []; DPSolutionPathCheck = []
    IndykTime = 0; DPTime = 0; ApprTime = 0
    DPObjValComponent = []; DPObjValPath = []
    
    TourSolutionPath = []; TourEdgePath = []; TourSolutionPathCheck = []
    TourEdgePathsTotal = []; 
    TourTime = 0
    TourObjValComponent = []; TourObjValPath = []
    
    Graph = nx.Graph(OriginalGraph)
    node_prizes = pcp.getPrizes(Graph)
    node_list = Graph.nodes(data=False)
    intersection_coordinates = rn.createInterDictionary(intersection_file)
    graph_nodes = Graph.nodes(data=True)
    (index_dictionary, node_dictionary, edges_dictionary, edges, prizes, costs) = expu.prepareFastPCST(Graph,lamda)

    start_time_Indyk = time.time()   
    ########## - Counting time for kTrees
    start_time_ktrees = time.clock()
    vertices, edges = pc.pcst_fast(edges, prizes, costs, -1, k, pruning, 0)
    end_time_ktrees = time.clock() - start_time_ktrees
    
    ########## 
    
    # print 'Time to create all subgraphs:',end_time
    IndykTime = time.time() - start_time_Indyk
    
    (dec_nodes, decoded_edges) = expu.decodeFast(index_dictionary,edges_dictionary,vertices,edges)
    start_time = time.time()
    decoded_nodes = []
    decoded_nodes.append(dec_nodes) 
    if len(decoded_edges) > 0:
        Graph = pcp.createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)
    # print 'Time to create induced graph:',time.time() - start_time
    to_plot = []
    start_time = time.time() 
    sub_graphs = nx.connected_component_subgraphs(Graph)
    for i, sub_graph in enumerate(sub_graphs):    
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)
        IndykSolutionComponent.append(sub_graph.nodes())
        to_plot.append(nodes)
        remove = list(set(node_list) - set(nodes))
        prize = pcstg.getPrize(remove,OriginalGraph)
        cost = pcstg.getCostEdges(OriginalGraph,edges)
        objective_value = prize + cost
        IndykObjValComponent.append(objective_value)
        
    # print 'Time to compute objective value of component:',time.time() - start_time
    
    # lamda = 1
    sub_graphs = nx.connected_component_subgraphs(Graph)
    print 'Beginning DP Path on Indyk ~~!!!'
    start_time = time.time()
    for i, sub_graph in enumerate(sub_graphs):
        # print 'Subgraph edges Path:',sub_graph.edges(data=False)
        # print
        # print 'Subgraph nodes Path:',sub_graph.nodes(data=False)
        # print
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
        # print 'Time to get total prize:',time.time()-start_time2
        start_time_DP = time.time()
        (min_path_DP, min_edges_DP, min_obj_value_DP, elapsed_time_DP) = \
                           pcp.runDPPC(sub_graph,OriginalGraph,total_prize,1)
        # print 'Min edges DP:',min_edges_DP
        # print 
        # print
        DPTime = time.time() - start_time_DP 
        # print 'Time for one subgraph:',DPTime
        DPSolutionPath.append(min_path_DP)
        DPEdgePath.append(min_edges_DP)
        IndykEdgeComponent.append(sub_graph.edges())
        DPObjValPath.append(min_obj_value_DP)
    
    tree_penalties = computeTreePenalties(IndykSolutionComponent,OriginalGraph)
    tree_costs = computeTreeEdges(IndykEdgeComponent,OriginalGraph)
    
    end_time_DP = time.time() - start_time
    print 'Time to run Path for all subgraphs:',end_time_DP
    
    # lamda = 1
    sub_graphs = nx.connected_component_subgraphs(Graph)
    print 'Beginning Tour on Indyk ~~!!!'
    
    start_time = time.time()
    ########## - Counting time for FindTour Total
    start_time_find_tour_total = time.clock()
    for i, sub_graph in enumerate(sub_graphs):
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
        # print 'Time to get total prize:',time.time()-start_time2
        start_time_DP = time.time()
        (min_path_Appr, min_edges_Appr, min_edges_Appr_Total, min_obj_value_Appr,end_time_find_tour_pc) = \
                           pcpa.runAppr(sub_graph,OriginalGraph,total_prize,1)
        # print 'Min Path Approximate:',min_path_Appr
        TourTime = time.time() - start_time_DP 
        TourSolutionPath.append(min_path_Appr)
        TourEdgePathsTotal.append(min_edges_Appr_Total)
        TourEdgePath.append(min_edges_Appr)
        TourObjValPath.append(min_obj_value_Appr)
    
    end_time_find_tour_total = time.clock() - start_time_find_tour_total
    
    ########## 
    
    end_time_AP = time.time() - start_time
    print 'Time to run Tour for all subgraphs:',end_time_AP
    # print
    # file_plot_path = directory_plots+str(city)+"IP.html"
    if pruning == "strong":
        file_plot_path = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykPaths.html'
        file_plot_component = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykTrees.html'
        file_plot_path_img = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykPaths.png'
        file_plot_component_img = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykTrees.png'
        # plotPath(DPSolutionPath,file_plot_path,file_plot_path_img,intersection_coordinates,node_prizes)
        # plotMap(to_plot,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes)
    # else:
        # file_plot_path = directory_plots+'L'+str(lamda)+'K'+str(k)+"NAIP.html"
        # file_plot_component = directory_plots+'K'+str(k)+"NAIC.html"
    
    # plotMap(DPSolutionPath,file_plot_path,intersection_coordinates,node_prizes)
    # plotMap(to_plot,file_plot_component,intersection_coordinates,node_prizes)

    return Graph,end_time_DP,DPTime,IndykSolutionComponent,DPSolutionPath,\
            IndykEdgeComponent,DPEdgePath,IndykObjValComponent,DPObjValPath,\
            end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath,\
            tree_penalties,tree_costs,end_time_ktrees,end_time_find_tour_total,end_time_find_tour_pc

def runGreedy(OriginalGraph,lamda,k,directory_plots,city,intersection_file):
    GreedyObjValComponent = []
    GreedySolutionComponent = []; GreedyEdgeComponent = []
    DPSolutionPath = []; DPEdgePath = []; DPSolutionPathCheck = []
    IndykTime = 0; DPTime = 0; ApprTime = 0
    DPObjValComponent = []; DPObjValPath = []
    
    TourSolutionPath = []; TourEdgePath = []; TourSolutionPathCheck = []
    TourEdgePathsTotal = []; 
    TourTime = 0
    TourObjValComponent = []; TourObjValPath = []
    
    Graph = nx.Graph(OriginalGraph)
    node_prizes = pcp.getPrizes(OriginalGraph)
    node_list = OriginalGraph.nodes(data=False)
    intersection_coordinates = rn.createInterDictionary(intersection_file)
    
    start_time_Greedy = time.time()
    (dec_nodes,decoded_edges,min_totscore) = pcstg.PCST(OriginalGraph,lamda,k)
    GreedyTime = time.time() - start_time_Greedy
    
    decoded_nodes = []
    decoded_nodes.append(dec_nodes)
    
    if len(decoded_edges) > 0:
        Graph = pcp.createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)
    
    to_plot = [] 
    sub_graphs = nx.connected_component_subgraphs(Graph)
    for i, sub_graph in enumerate(sub_graphs):
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)
        GreedySolutionComponent.append(sub_graph.nodes())
        to_plot.append(nodes)
        remove = list(set(node_list) - set(nodes))
        prize = pcstg.getPrize(remove,OriginalGraph)
        cost = pcstg.getCostEdges(OriginalGraph,edges)
        objective_value = prize + cost
        GreedyObjValComponent.append(objective_value)
    
    # lamda = 1
    sub_graphs = nx.connected_component_subgraphs(Graph)
    print 'Beginning DP Path on Greedy ~~!!!'
    start_time = time.time()
    for i, sub_graph in enumerate(sub_graphs):
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
        # print 'Time to get total prize:',time.time()-start_time2
        start_time_DP = time.time()
        (min_path_DP, min_edges_DP, min_obj_value_DP, elapsed_time_DP) = \
                           pcp.runDPPC(sub_graph,OriginalGraph,total_prize,1)
        DPTime = time.time() - start_time_DP 
        # print 'Time for one subgraph:',DPTime
        DPSolutionPath.append(min_path_DP)
        DPEdgePath.append(min_edges_DP)
        GreedyEdgeComponent.append(sub_graph.edges())
        DPObjValPath.append(min_obj_value_DP)
    
    end_time_DP = time.time() - start_time
    # print 'Time to runDPPC for all subgraphs:',end_time_DP
    
    # lamda = 1
    sub_graphs = nx.connected_component_subgraphs(Graph)
    print 'Beginning Tour on Greedy~~!!!'
    start_time = time.time()
    for i, sub_graph in enumerate(sub_graphs):
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
        # print 'Time to get total prize:',time.time()-start_time2
        start_time_DP = time.time()
        (min_path_Appr, min_edges_Appr, min_edges_Appr_Total, min_obj_value_Appr, elapsed_time_Appr) = \
                           pcpa.runAppr(sub_graph,OriginalGraph,total_prize,1)
        TourTime = time.time() - start_time_DP 
        TourSolutionPath.append(min_path_Appr)
        TourEdgePathsTotal.append(min_edges_Appr_Total)
        TourEdgePath.append(min_edges_Appr)
        TourObjValPath.append(min_obj_value_Appr)
        
    end_time_AP = time.time() - start_time
    print 'Time to runAppr for all subgraphs:',end_time_AP
    file_plot_path = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'GreedyPaths.html'
    file_plot_component = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'GreedyTrees.html'
    file_plot_path_img = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'GreedyPaths.png'
    file_plot_component_img = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'GreedyTrees.png'
    # plotPath(DPSolutionPath,file_plot_path,file_plot_path_img,intersection_coordinates,node_prizes)
    # plotMap(to_plot,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes)

    return Graph,end_time_DP,DPTime,GreedySolutionComponent,DPSolutionPath,\
            GreedyEdgeComponent,DPEdgePath,GreedyObjValComponent,DPObjValPath,\
            end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath
        
def createInterDictionary(filename):
    intersection_coord = {} 
    reader = csv.reader(open(filename, 'r'))
    counter = 0
    for row in reader:
        key, value = row     
        intersection_coord[key] = value     
        
    return intersection_coord

    
def createInducedSubgraph(decoded_edges,dec_nodes):
    Graph = nx.Graph()
    # print 'Decoded edges:',decoded_edges
    for component in decoded_edges:
        # print 'C',component
        for edge in component:
            u = edge[0]; v = edge[1]
            Graph.add_edge(u,v)
    
    for node in dec_nodes:
        if not Graph.has_node(node):
            # print 'node:',node
            Graph.add_node(node)
    return Graph


def kDP(subgraphs,OrigGraph,lamda,k,intersection_coordinates):
    inf = float("inf")
    node_values = OrigGraph.nodes(data=True)

    DPSolutionPath = []; DPSolutionEdges = []
    DPSolutionObjVal = []
    file_plot_component = 'BostonTwitterComp'+str(k)+'L'+str(lamda)+'kDP.html'
    file_plot_path = 'PathBostonTwitterComp'+str(k)+'L'+str(lamda)+'kDP.html'
    solution_paths = []; solution_edges = []; solution_obj_values = []
    min_paths = []; min_edges = []
    GreedyMatrix = {}; GreedyMatrixNodes = {};  GreedyMatrixEdges = {}
    counter = 0
    node_prizes = getPrizes(OrigGraph)
    tot_prize = ptp.computeObjectiveValueGraph(OrigGraph)
    LamdaGraph = nx.Graph(OrigGraph)
    MSTS = []
    to_plot = []
    # for each subgraph
    for i,MST in enumerate(subgraphs):
        LamdaGraph = nx.Graph(OrigGraph)
        
        PCGraph = nx.Graph(MST)
        # print 'Length PCGRaph:',len(PCGraph.nodes(data=False))
        
        GreedyMatrix[i+1] = {}
        GreedyMatrixNodes[i+1] = {}
        GreedyMatrixEdges[i+1] = {}
        # print 'Nodes MST:',MST.nodes(data=False)
        # to_plot = []
        nodes = MST.nodes(data=False)
        edges = MST.edges(data=False)
        MSTS.append(edges)
        # print 'MST edges:',MST.edges(data=True)
        # print
        to_plot.append(nodes)
        # print 'To plot:',to_plot
        # node_prizes = getPrizesFromOriginal(MST,OrigGraph)
        # node_prizes = getPrizes(MST)
        objective_value = tot_prize 
        # remember previous objective value
        prev_objective = objective_value
        # greedy -- for path 0
        solution_obj_values.append(0)
        GreedyMatrix[i+1][0] = 0
        GreedyMatrixNodes[i+1][0] = []
        GreedyMatrixEdges[i+1][0] = []
         
        # print 'Number of nodes:',len(nodes)
        while (counter < k):
            # print '1 counter',counter,'k',k
            # print '1 Filling positions[',i+1,'][',counter+1,']'
            TreeGraph = nx.Graph(PCGraph)
            LamdaGraph = nx.Graph(OrigGraph)
            if (counter > nx.number_of_nodes(MST)):
                # print '1 In this case'
                solution_obj_values.append(prev_objective)
                GreedyMatrix[i+1][counter+1] = inf # prev_objective
                GreedyMatrixNodes[i+1][counter+1] = []
                GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                GreedyMatrixEdges[i+1][counter+1] = []
                GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
            else:   
                # print '2 In this case'
                # print 'Length TreeGraph:',len(TreeGraph.nodes(data=False))
                tree_nodes = TreeGraph.nodes(data=False)
                num_tree_nodes = len(tree_nodes)
                if num_tree_nodes == 1:
                    # print '3 In this case'
                    if counter+1 == 1:
                        # print '4 In this case'
                        nd = tree_nodes[0]
                        prize = OrigGraph.node[nd]['prize']
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
                        # print '5 In this case'
                        solution_obj_values.append(prev_objective)
                        GreedyMatrix[i+1][counter+1] = inf # prev_objective
                        GreedyMatrixNodes[i+1][counter+1] = []
                        GreedyMatrixNodes[i+1][counter+1] = GreedyMatrixNodes[i+1][counter+1] + min_paths
                        GreedyMatrixEdges[i+1][counter+1] = []
                        GreedyMatrixEdges[i+1][counter+1] = GreedyMatrixEdges[i+1][counter+1] + min_edges
                    
                else:   

                    nodes_lamda = LamdaGraph.nodes(data=True)

                    (min_paths,min_edges,min_obj_value,end_time_find_binarize_pc,end_time_find_paths_on_tree_pc,\
                                       end_time_retrieve_path_pc) = ptp.runKPCSP(LamdaGraph,TreeGraph,counter+1)
   
                    if not min_edges:
                        nd = min_paths[0]
                        prize = OrigGraph.node[nd[0]]['prize']
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
                            ptp.computeObjectiveValue(LamdaGraph,min_paths,min_edges,False)

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
            = ptp.runGreedyDP(GreedyMatrix,GreedyMatrixNodes,GreedyMatrixEdges,tot_prize,k)
    # print 'tototot plot',to_plot
    # plotMap(to_plot,file_plot_component,intersection_coordinates,node_prizes)
    # plotPath(DPSolutionPath,file_plot_path,intersection_coordinates,node_prizes) 
    # print 'Returning DP Solution Path:',DPSolutionPath
    # print 'Returning DP Solution Edges:',DPSolutionEdges
    # print 'Returning DP score:',opt_score
    return DPSolutionPath,DPSolutionEdges,opt_score

def runDensityClustering(LamdaGraph,intersection_file,solution_components,edge_components,\
                 lats,lons,directory_plots,city,K,lamda,flag,algorithm):

    DBSCANObjValComponent = []
    DBSCANSolutionComponent = []; DBSCANEdgeComponent = []
    DPSolutionPath = []; DPEdgePath = []; DPSolutionPathCheck = []
    kDPSolutionPath = []; kDPEdgePath = []; kDPSolutionPathCheck = []
    
    DBSCANTime = 0; DPTime = 0; ApprTime = 0
    DPObjValComponent = []; DPObjValPath = []
    kDPObjValComponent = []; kDPObjValPath = []
    
    TourSolutionPath = []; TourEdgePath = []; TourSolutionPathCheck = []
    TourEdgePathsTotal = []; 
    TourTime = 0
    TourObjValComponent = []; TourObjValPath = []
    
    Graph = nx.Graph(LamdaGraph)
    node_prizes = pcp.getPrizes(Graph)
    node_list = Graph.nodes(data=False)
    intersection_coordinates = rn.createInterDictionary(intersection_file)
    graph_nodes = Graph.nodes(data=True)

    dec_nodes = [item for sublist in solution_components for item in sublist]
    decoded_edges = edge_components
    
    start_time = time.time()
    decoded_nodes = []
    decoded_nodes.append(dec_nodes)
    
    if len(decoded_edges) > 0:
        Graph = createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)

    to_plot = []
    sizes = []
    tree_nodes = []
    start_time = time.time() 
    sub_graphs = nx.connected_component_subgraphs(Graph)
    # print 'Number of connected subgraphs:',len(list(sub_graphs))
    count_appends = 1;
    for i, sub_graph in enumerate(sub_graphs):
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)
        # print 'Length of nodes:',len(nodes)
        # print 'Appending nodes:',nodes
        DBSCANSolutionComponent.append(nodes)
        to_plot.append(nodes)
        remove = list(set(node_list) - set(nodes))
        prize = pcstg.getPrize(remove,LamdaGraph)
        # print 'edges:',edges
        cost = pcstg.getCostEdges(LamdaGraph,edges)
        objective_value = prize + cost
        count_appends+=1
        DBSCANObjValComponent.append(objective_value)
    
    for components in DBSCANSolutionComponent:
        tree_nodes = tree_nodes + components
        
    
    subgraphs = nx.connected_component_subgraphs(Graph)
    kDPSolutionPath,kDPSolutionEdges,kopt_score = kDP(subgraphs,LamdaGraph,lamda,K,intersection_coordinates)
    sub_graphs = nx.connected_component_subgraphs(Graph)
    # print 'Beginning DP Path on Indyk ~~!!!'
    start_time = time.time()
    for i, sub_graph in enumerate(sub_graphs):
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
    
        start_time_DP = time.time()
        # print '3 Length subgraph:',len(sub_graph.nodes(data=False))
        (min_path_DP,min_edges_DP,min_obj_value_DP,end_time_find_binarize_pc,end_time_find_paths_on_tree_pc,\
                                       end_time_retrieve_path_pc) = ptp.runKPCSP(LamdaGraph,sub_graph,1)
        
        DPTime = time.time() - start_time_DP 
        # print 'Appending nodes:',min_path_DP[0]
        DPSolutionPath.append(min_path_DP[0])
        DPEdgePath.append(min_edges_DP[0])
        DBSCANEdgeComponent.append(sub_graph.edges())
        DPObjValPath.append(min_obj_value_DP)
        
    # print '1 DP Solution Path:',min_path_DP
    # print '1 DP Solution Edges:',min_edges_DP
    # print '1 DP score:',min_obj_value_DP
    
    tree_penalties = computeTreePenalties(DBSCANSolutionComponent,LamdaGraph)
    tree_costs = computeTreeEdges(DBSCANEdgeComponent,LamdaGraph)
    end_time_DP = time.time() - start_time

    sub_graphs = nx.connected_component_subgraphs(Graph)
    # print 'Beginning Tour on Indyk ~~!!!'
    start_time = time.time()
    for i, sub_graph in enumerate(sub_graphs):
        edges = sub_graph.edges(data=False)
        nodes = sub_graph.nodes(data=False)      
        start_time2 = time.time()
        total_prize = pcp.getTotalPrize(node_list,sub_graph,node_prizes)
        # print 'Time to get total prize:',time.time()-start_time2
        start_time_DP = time.time()
        (min_path_Appr, min_edges_Appr, min_edges_Appr_Total, min_obj_value_Appr, elapsed_time_Appr) = \
                            pcpa.runAppr(sub_graph,LamdaGraph,total_prize,1)
        TourTime = time.time() - start_time_DP 
        TourSolutionPath.append(min_path_Appr)
        TourEdgePathsTotal.append(min_edges_Appr_Total)
        TourEdgePath.append(min_edges_Appr)
        TourObjValPath.append(min_obj_value_Appr)
    
    end_time_AP = time.time() - start_time
    if flag == True:
        # file_plot_path = directory_plots+'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Paths.html'
        # file_plot_component = directory_plots+'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Trees.html'
        # file_plot_path_img = directory_plots+'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Paths.png'
        # file_plot_component_img = directory_plots+'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Trees.png'
        file_plot_path = 'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Paths.html'
        file_plot_component = 'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Trees.html'
        file_plot_path_img = 'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Paths.png'
        file_plot_component_img = 'TBK'+str(K)+'L'+str(lamda)+str(algorithm)+'Trees.png'
        plotPath(DPSolutionPath,file_plot_path,file_plot_path_img,intersection_coordinates,node_prizes)
        plotMap(DBSCANSolutionComponent,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes)
    
    return Graph,end_time_DP,DPTime,DBSCANSolutionComponent,DPSolutionPath,\
            DBSCANEdgeComponent,DPEdgePath,DBSCANObjValComponent,DPObjValPath,\
            end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath,tree_nodes,\
            tree_penalties,tree_costs,kDPSolutionPath,kDPSolutionEdges,kopt_score 
        
def applyLamda(GraphInit,lamda):  
    Graph = nx.Graph(GraphInit)
    for node in Graph.nodes(data=False):
        # print 'old value:',Graph.node[node]['prize']
        Graph.node[node]['prize'] = Graph.node[node]['prize']*(lamda)
        # print 'new value:',Graph.node[node]['prize']
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


def computeTotalCost(OriginalGraph,IndykEdgeComponent):
    cost = 0
    counter = 0
    # print OriginalGraph.edges(data=False)
    # print
    for edges in IndykEdgeComponent: 
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
                print 'Oh noooo edge:',edge
    # print 'Length of edges:',counter        
    # print
    return cost
                                                                                                                        
def computeObjectiveValue(OriginalGraph,IndykSolutionComponent,IndykEdgeComponent,print_output):
    tot_nodes = OriginalGraph.nodes(data=False)

    length_input = 0
    component_nodes = []

    for component in IndykSolutionComponent:
        # print 'Component:',component
        component_nodes = component_nodes + component
        length_input+=len(component)
    
    penalty_nodes = list(set(tot_nodes)^set(component_nodes))
    
    obj_prize = computeTotalPrize(OriginalGraph,penalty_nodes)

    obj_cost = computeTotalCost(OriginalGraph,IndykEdgeComponent)

    objective_value = obj_prize + obj_cost
    if print_output:
        print 'Final penalty is:',obj_prize
        print 'Final cost is:',obj_cost
    return objective_value,obj_prize,obj_cost

def computeDiffTreePath(OrigGraph,tree_nodes,path_nodes):

    penalty_nodes = list(set(tree_nodes)^set(path_nodes))
    
    penalty = computeTotalPrize(OrigGraph,penalty_nodes)
    
    return penalty
def createEdgesFromDPPath(paths):
    edges = []; final_edges = []; counter = 1
    # print 'Create edges from paths:',paths
    for path in paths:
        edges = []
        if len(path) == 1:
            final_edges.append(edges)
        else:
            # print '********* Path is *************',path
            while counter < len(path):
                u = path[counter-1]; v = path[counter]
                edge = (u,v)
                # print 'Appending edge:',edge,'prev index:',counter-1,'current index',counter
                edges.append(edge)
                counter+=1
                
        final_edges.append(edges)
        counter = 1
    
    # print 'Final edges:',final_edges
        
    return final_edges
                 
        
def kBinarizedPCSP(LamdaGraph,OriginalGraph,lamda,k,intersection_coordinates,pruning):
    # node_prizes = getPrizes(LamdaGraph)
    file_plot_path = 'PathBostonTwitterComp'+str(k)+'L'+str(lamda)+'Binarized.html'
    file_plot_path_img = 'PathBostonTwitterComp'+str(k)+'L'+str(lamda)+'BinarizedImg.png'
    file_plot_component = 'BostonTwitterComp'+str(k)+'L'+str(lamda)+'Binarized.html'
    file_plot_component_img = 'BostonTwitterComp'+str(k)+'L'+str(lamda)+'BinarizedImg.png' 
    Graph = nx.Graph(OriginalGraph)
    LGraph = nx.Graph(LamdaGraph)

    node_prizes = pcp.getPrizes(Graph)
    node_list = Graph.nodes(data=False)

    graph_nodes = Graph.nodes(data=True)

    (index_dictionary, node_dictionary, edges_dictionary, edges, prizes, costs) = expu.prepareFastPCST(LGraph,lamda)

    start_time_Indyk = time.time()    
    start_time = time.time()
    print 'K:',k

    vertices, edges = pc.pcst_fast(edges, prizes, costs, -1, 1, pruning, 0)

    end_time = time.time() - start_time

    IndykTime = time.time() - start_time_Indyk
     
    (dec_nodes, decoded_edges) = expu.decodeFast(index_dictionary,edges_dictionary,vertices,edges)
    start_time = time.time()
    decoded_nodes = []
    decoded_nodes.append(dec_nodes) 
    if len(decoded_edges) > 0:
        Graph = pcp.createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)
 
    to_plot = []
    start_time = time.time() 
    sub_graphs = nx.connected_component_subgraphs(Graph)
     
    for i, sub_graph in enumerate(sub_graphs):    
        node_list = sub_graph.nodes(data=False)
        to_plot.append(node_list)
        print 'Length of tree in 1:',len(node_list)
        # plotMap(to_plot,file_plot_component,intersection_coordinates,node_prizes)
        root = node_list[0]
        # print 'Number of nodes in tree:',len(node_list)
        (objective_value,final_paths,D,Dp,H,Hp,L,Lp,B,Bp,end_time_find_binarize_pc,end_time_find_paths_on_tree_pc,\
                                       end_time_retrieve_path_pc) = bt.kBinarizedPCSP(LGraph, sub_graph, root, k)
        
    # print 'Final paths:',final_paths
    # print 'Length of paths after binarized function:',len(final_paths)
    # print 'Paths are after binarized function:',final_paths 
    # print 'Graph edges:',Graph.edges(data=False)
    # print 'Final Paths:',final_paths
    final_edges = createEdgesFromDPPath(final_paths)
    # print 'Final edges:',final_edges
    # to_plot = LGraph.nodes(data=False)
    # print 'Input nodes for DP Length to plot:',len(to_plot)
    
    # plotMap(to_plot,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes) 
    # plotPath(final_paths,file_plot_path,file_plot_path_img,intersection_coordinates,node_prizes) 
    # return final_paths, final_edges, objective_value 
    return final_paths, final_edges, objective_value, Graph 

def PCSP_GreedyPath(LamdaGraph,OriginalGraph,lamda,k,intersection_coordinates,pruning):
    # node_prizes = getPrizes(LamdaGraph)
    file_plot_path = 'BostonTwitter'+str(k)+'L'+str(lamda)+'kPCSP.html'
    file_plot_component = 'BostonTwitterComp'+str(k)+'L'+str(lamda)+'kPCSP.html'
    
    Graph = nx.Graph(OriginalGraph)
    LGraph = nx.Graph(LamdaGraph)

    node_prizes = pcp.getPrizes(Graph)
    node_list = Graph.nodes(data=False)

    graph_nodes = Graph.nodes(data=True)
    node_feat = Graph.nodes(data=True)
    node_lfeat = LGraph.nodes(data=True)

    DPSolutionPath = []
    DPSolutionEdges = []
    DPSolutionObjVal = []
    
    (index_dictionary, node_dictionary, edges_dictionary, edges, prizes, costs) = expu.prepareFastPCST(LGraph,lamda)

    start_time_Indyk = time.time()    
    start_time = time.time()
    print 'K:',k

    vertices, edges = pc.pcst_fast(edges, prizes, costs, -1, 1, pruning, 0)

    end_time = time.time() - start_time

    IndykTime = time.time() - start_time_Indyk
     
    (dec_nodes, decoded_edges) = expu.decodeFast(index_dictionary,edges_dictionary,vertices,edges)
    start_time = time.time()
    decoded_nodes = []
    decoded_nodes.append(dec_nodes) 
    if len(decoded_edges) > 0:
        Graph = pcp.createInducedSubgraph(decoded_edges,dec_nodes)
    else:
        node_list = Graph.nodes()
        node_remains = decoded_nodes[0]
        remove = list(set(node_list) - set(node_remains))
        Graph.remove_nodes_from(remove)
 
    to_plot = []
    start_time = time.time() 
    sub_graphs = nx.connected_component_subgraphs(Graph)
     
    counter = 0
    for i, MST in enumerate(sub_graphs):    
        node_list = MST.nodes(data=False)
        to_plot.append(node_list)
        root = node_list[0]
        while (counter < k):
            print 'Greedy DP length MST:',len(MST.nodes(data=False))
            (min_path,min_edges,min_obj_value) = ptp.runPCSP(MST,LamdaGraph,OriginalGraph,lamda,k,intersection_coordinates)
                
            # (min_path,min_edges,min_obj_value) = runKPCSP(Graph,OrigGraph,OriginalGraph,1)
                
            # print 'in greedy DP Objective path:',min_obj_value
            # print 'in greedy DP Min path:',min_path 
            if isinstance(min_path, int):
                MST.remove_node(min_path)
            else:
                for node in min_path:
                    MST.remove_node(node)
                 
            # print 'New number of nodes in MST:',len(MST.nodes(data=False))
            DPSolutionPath.append(min_path)
            DPSolutionEdges.append(min_edges)
            DPSolutionObjVal.append(min_obj_value)
            # print 'min objective value:',min_obj_value
            counter+=1
            
            
    objective_value,penalty,cost = computeObjectiveValue(LamdaGraph,DPSolutionPath,DPSolutionEdges,False)
        
    # plotPath(DPSolutionPath,file_plot_path,intersection_coordinates,node_prizes) 
    return DPSolutionPath, DPSolutionEdges, objective_value 

def findCompAboveThreshold(OriginalGraph,city,intersection_coordinates,thresh):
    file_plot_component = city+'Map.html'
    file_plot_component_img = city+'Map.png'
    total_nodes = 0
    max_num_nodes = -float('inf')
    Graph = nx.Graph(OriginalGraph)
    node_prizes = pcp.getPrizes(Graph)
    # node threshold
    sub_graphs = nx.connected_component_subgraphs(Graph)
    to_plot = []
    for i, sub_graph in enumerate(sub_graphs):
        node_list = sub_graph.nodes(data=False)
        num_nodes = len(sub_graph.nodes(data=False))
        total_nodes += num_nodes
        print 'Number of nodes in component',num_nodes
        if num_nodes >= thresh:
            # to_plot = [] 
            to_plot.append(node_list)
            max_num_nodes = num_nodes
            print 'New maximum number of nodes:',max_num_nodes
            BiggestComponent = nx.Graph(sub_graph)
    
    print 'Total number of nodes for',city,'is:',total_nodes
    # plotMap(to_plot,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes)
    return BiggestComponent

def findBiggestComponent(OriginalGraph,city,intersection_coordinates):
    file_plot_component = city+'Map.html'
    file_plot_component_img = city+'Map.png'
    total_nodes = 0
    max_num_nodes = -float('inf')
    Graph = nx.Graph(OriginalGraph)
    node_prizes = pcp.getPrizes(Graph)
    # node threshold
    thresh = 100
    sub_graphs = nx.connected_component_subgraphs(Graph)
    to_plot = []
    for i, sub_graph in enumerate(sub_graphs):
        node_list = sub_graph.nodes(data=False)
        num_nodes = len(sub_graph.nodes(data=False))
        total_nodes += num_nodes
        # print 'Number of nodes in component',num_nodes
        if num_nodes >= max_num_nodes:
            BiggestComponent = nx.Graph(sub_graph)
            max_num_nodes = num_nodes
    
    # print 'Total number of nodes for',city,'is:',total_nodes
    # plotMap(to_plot,file_plot_component,file_plot_component_img,intersection_coordinates,node_prizes)
    return BiggestComponent



def create_csv(points, file_out,intersection_coordinates,OriginalGraph,path):
    colors = ["#0086b3", "#b34700", "#008000","#b38600", "#8600b3", "#0000b3",
              "#00b300", "#00b386", "#b3b300", "#b30086"]
    
    DF_Sol = pd.DataFrame()
    lats = []
    longs = []
    limit = len(points)
    lat_lon_pairs = []
    counter = 0
    prev = float("inf")
    features_list = []
    if path == False:
        print 'in hereeee'
        lat_lon_pair = []
        if len(points) == 1:
            Graph = nx.Graph(OriginalGraph)
            H = Graph.subgraph(points[0])
            edges = list(nx.dfs_edges(H))
            for u, v, data in nx.dfs_labeled_edges(H):
                coord = intersection_coordinates[str(u)]
                lat_long = [x.strip() for x in coord.split(',')]
                lat = float(lat_long[0])
                lon = float(lat_long[1])
                tup1 = (lon,lat)
                lat_lon_pairs.append(tup1)
                coord = intersection_coordinates[str(v)]
                lat_long = [x.strip() for x in coord.split(',')]
                lat = float(lat_long[0])
                lon = float(lat_long[1])
                tup2 = (lon,lat)
                lat_lon_pairs.append(tup2)
                lat_lon_pairs.append(tup2)
                lat_lon_pairs.append(tup1)
                
                
                node_color = colors[counter]

            di = {}           
            path_name = "path"+str(counter)
            di["number"] = path_name
            counter+=1
            my_line = LineString(coordinates=lat_lon_pairs)
            print 'Appending:',my_line
            features_list.append(Feature(geometry=my_line,properties=di))
            lat_lon_pairs = []
            
        else:
            for point in points:
                if isinstance(point,int):
                    coord = intersection_coordinates[str(point)]
                    lat_long = [x.strip() for x in coord.split(',')]
                    lat = float(lat_long[0])
                    lon = float(lat_long[1])
                    tup = (lon,lat)
                    lat_lon_pairs.append(tup)
                    lats.append(lat)
                    longs.append(lon)
                    tup = (lat,lon)
                    lat_lon_pairs.append(tup)
                    tup2 = (lat+0.0005,lon+0.0005)
                    lat_lon_pairs.append(tup2)
                else:
                    Graph = nx.Graph(OriginalGraph)
                    H = Graph.subgraph(point)

                    edges = list(nx.dfs_edges(H))

                    for u, v, data in nx.dfs_labeled_edges(H):
                        coord = intersection_coordinates[str(u)]
                        lat_long = [x.strip() for x in coord.split(',')]
                        lat = float(lat_long[0])
                        lon = float(lat_long[1])
                        tup1 = (lon,lat)
                        lat_lon_pairs.append(tup1)
                        coord = intersection_coordinates[str(v)]
                        lat_long = [x.strip() for x in coord.split(',')]
                        lat = float(lat_long[0])
                        lon = float(lat_long[1])
                        tup2 = (lon,lat)
                        lat_lon_pairs.append(tup2)
                        lat_lon_pairs.append(tup2)
                        lat_lon_pairs.append(tup1)
                        
                
                if len(list(set(lat_lon_pairs))) == 1:
                    tup = lat_lon_pairs[0]
                    tup2 = (tup[0]+0.0005,tup[1]+0.0005)
                    lat_lon_pairs.append(tup2)
                    
                di = {}           
                path_name = "path"+str(counter)
                di["number"] = path_name
                counter+=1
                
                my_line = LineString(lat_lon_pairs)
                
                print '1 Appending:',my_line
                features_list.append(Feature(geometry=my_line,properties=di))
                lat_lon_pairs = []
        
    elif path == True:
        lats = []
        longs = []
        limit = len(points)
        lat_lon_pairs = []
        lat_lon_pair = []
        if isinstance(points[0],int):
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
                        tup = (lon,lat)
                        lat_lon_pairs.append(tup)
                        lats.append(lat)
                        longs.append(lon)
                else:
                    coord = intersection_coordinates[str(nodes[0])]
                    lat_long = [x.strip() for x in coord.split(',')]
                    lat = float(lat_long[0])
                    lon = float(lat_long[1])
                    tup = (lon,lat)
                    lat_lon_pairs.append(tup)
                    lats.append(lat); 
                    longs.append(lon); 
                    
            di = {}           
            path_name = "path"+str(counter)
            di["number"] = path_name
            counter+=1

            my_line = LineString(lat_lon_pairs)

            print '1 Appending:',my_line
            features_list.append(Feature(geometry=my_line,properties=di))
            lat_lon_pairs = []
            
        else:
            print 'points'
            for point in points:
                lat_lon_pair = []
                print 'P:',point
                if isinstance(point,int):
                    coord = intersection_coordinates[str(point)]
                    lat_long = [x.strip() for x in coord.split(',')]
                    lat = float(lat_long[0])
                    lon = float(lat_long[1])
                    tup = (lon,lat)
                    lat_lon_pairs.append(tup)
                    lats.append(lat)
                    longs.append(lon)
                    tup = (lat,lon)
                    lat_lon_pairs.append(tup)
                    tup2 = (lat+0.0002,lon+0.0002)
                    lat_lon_pairs.append(tup2)
                else:
                    for nds in point:
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
                                tup = (lon,lat)
                                lat_lon_pairs.append(tup)
                                lats.append(lat)
                                longs.append(lon)
                        else:
                            coord = intersection_coordinates[str(nodes[0])]
                            lat_long = [x.strip() for x in coord.split(',')]
                            lat = float(lat_long[0])
                            lon = float(lat_long[1])
                            tup = (lon,lat)
                            lat_lon_pairs.append(tup)
                            lats.append(lat); 
                            longs.append(lon);
                            
                    if len(list(set(lat_lon_pairs))) == 1:
                        tup = lat_lon_pairs[0]
                        tup2 = (tup[0]+0.0002,tup[1]+0.0002)
                        lat_lon_pairs.append(tup2)
                        
                    di = {}           
                    path_name = "path"+str(counter)
                    di["number"] = path_name
                    counter+=1

                    my_line = LineString(lat_lon_pairs)

                    print '1 Appending:',my_line
                    features_list.append(Feature(geometry=my_line,properties=di))
                    lat_lon_pairs = []
    
    
    my_feature_collection = FeatureCollection(features_list)
    
    with open(file_out, 'w') as outfile:
        geojson.dump(my_feature_collection, outfile)
    
    # print my_point
    
    DF_Sol['latitude'] = lats
    DF_Sol['longitude'] = longs
    
def main():
    min_r = 0; max_r = 1; samples = 1000; step=0.0001
    graph_file = str(sys.argv[1])
    intersections_file = str(sys.argv[2])
    # k = int(sys.argv[3]);
    # Boston Tweets: 1002839
    # Austin Tweets: 220588
    # San Francisco Tweets: 
    # flickr Boston: 300189
    # flickr Austin: 180900
    # flickr SF: 1115177
    # pruning = str(sys.argv[4]) 
    
    counter = 0 
    Component_Labels = {}
    labels = 0
    intersection_coordinates = rn.createInterDictionary(intersections_file)
    parameter_seed = 1
    DF_Experiment = pd.DataFrame()
    DF_Debug = pd.DataFrame()
    DF_Times = pd.DataFrame()
    
    # Experimental Results
    LamdaValues = []; KValues = []
    
    InTopObjVal = []; NPInTopObjVal = []; GrTopObjVal = []; DBSCANTopObjVal = []
    InTopObjValTour = []; NPInTopObjValTour = []; GrTopObjValTour = []; DBSCANTopObjValTour = []
    InTopObjValPath = []; NPInTopObjValPath = []; GrTopObjValPath = []; DBSCANTopObjValPath = []
    kDBSCANTopObjValPath = []
    
    InNodesComponent = []; GrNodesComponent = []; NPInNodesComponent = []; DBSCANNodesComponent = []
    InNodesPath = []; GrNodesPath = []; NPInNodesPath = []; DBSCANNodesPath = []; kDBSCANNodesPath = [];  
    InEdgesComponent = []; GrEdgesComponent = []; NPInEdgesComponent = []; DBSCANEdgesComponent = []; 
    InEdgesPath = []; GrEdgesPath =[]; NPInEdgesPath = []; DBSCANEdgesPath =[]; kDBSCANEdgesPath = []
    
    InTime = []; GrTime = []; NPInTime = []; DBSCANTime = [];
    InDPTime = []; GrDPTime = []; NPInDPTime = []; DBSCANDPTime = []
    InObjValComponent = []; GrObjValComponent = []; NPInObjValComponent = []; DBSCANObjValComponent = []
    InObjValPath = []; GrObjValPath = []; NPInObjValPath = []; DBSCANObjValPath = []; kDBSCANObjValPath = []
    
    ApNodesPath = []; ApEdgesPath = []; ApEdgesTotalPath = []; ApObjValPath = []
    ApTime = []; InAPTime = []
    
    GrApNodesPath = []; GrApEdgesPath = []; GrApEdgesTotalPath = []; GrApObjValPath = []
    GrApTime = []; GrInAPTime = []
    
    NPApNodesPath = []; NPApEdgesPath = []; NPApEdgesTotalPath = []; NPApObjValPath = []
    NPApTime = []; NPInAPTime = []
    
    DBSCANApNodesPath = []; DBSCANApEdgesPath = []; DBSCANApEdgesTotalPath = []; DBSCANApObjValPath = []
    DBSCANApTime = []; DBSCANInAPTime = []
    
    GMMTopObjVal = []; GMMTopObjValTour = []; GMMTopObjValPath = []
    GMMNodesComponent = []; GMMNodesPath = []; GMMEdgesComponent = []; GMMEdgesPath =[]
    GMMTime = []; GMMDPTime = []; GMMObjValComponent = []; GMMObjValPath = []
    GMMApNodesPath = []; GMMApEdgesPath = []; GMMApEdgesTotalPath = []; GMMApObjValPath = []
    GMMApTime = []; GMMInAPTime = []
    
    GRPTopObjValPath = []; GRPNodesPath = []; GRPEdgesPath =[]; GRPObjValPath = []
    GRDPTopObjValPath = []; GRDPNodesPath = []; GRDPEdgesPath =[]; GRDPObjValPath = []
    kDPTopObjValPath = []; kDPNodesPath = []; kDPEdgesPath =[]; kDPObjValPath = []
    kBestDPTopObjValPath = []; kBestDPNodesPath = []; kBestDPEdgesPath =[]; kBestDPObjValPath = []
    Nodes = []; Edges = []
    Cities = []
    
    # Debug
    VTIndyk = []; VTDBSCAN = []
    VPIndyk = []; VPDBSCAN = []
    CTIndyk = []; CTDBSCAN = []
    CPIndyk = []; CPDBSCAN = []
    
    kTreeTimes = []; FindTourTimesPC = []; FindTourTimesT = []; kToursTimes = []; BinarizationTimesT = []
    BinarizationTimesPC = []; FindPathsOnTreeTimesT = []; FindPathsOnTreeTimesPC = []; SplitBudgetTimesPC = []
    RetrievePathTimesT = []; RetrievePathTimesPC = [];
    FindPathsTimes = []
    
    DebugSumIndyk = []; DebugSumDBSCAN = []

    # print 'K is: ',k
    start_time = time.time() 
    directory = os.getcwd()
    # city = graph_file.replace("Graph.gml","")
    # city = city.replace(directory,"")
    city = "sf_twitter"
    print 'City:',city
    lamdas = []
    
    ks = []
    # ks = [2,4,10]
    for i in range(1,21,1):
        ks.append(i)

    # lamdas = [0.0001,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005]
    lamdas = [0.002]
    # lamdas = [0.0001,0.0005]
    # lamdas = [0.0025]

    directory_plots = directory+'/'+str(city)+'/'
    directory = directory_plots
    if not os.path.isdir(directory_plots):
        os.makedirs(directory_plots)
    ks = [5]
    
    start_time_all = time.time()
    for lamda in lamdas:
        for k in ks: 
            file_plot_path = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykPaths.html'
            file_plot_path_img = directory_plots+'TBK'+str(k)+'L'+str(lamda)+'IndykPaths.png'
            print 'Lamda is:',lamda
            LamdaValues.append(lamda)
            KValues.append(k)
            labels = 0
            Component_Labels = {}
            FileGraph = nx.read_gml(graph_file)
            # BiggestComponent = findCompAboveThreshold(OriginalGraph,city,intersection_coordinates)
            BiggestComponent = findBiggestComponent(FileGraph,city,intersection_coordinates)
            print 'Length number of components:',len(BiggestComponent)
            OriginalGraph = nx.Graph(BiggestComponent)
            
            print 'Size of largest component',len(BiggestComponent.nodes(data=False))
            # sys.exit()
            
            # print 'Original Graph Edges:',OriginalGraph.edges(data=False)
            # print 
            OrigGraph = applyLamda(OriginalGraph,lamda)
            
            total_prize = totalNode(OriginalGraph)
            total_edges = totalEdge(OriginalGraph)

            Graph = nx.Graph(OrigGraph)
            # print 'Graph nodes:',Graph.nodes()
            city = graph_file.replace("Graph.gml","")
            city = city.replace(directory,"")

            node_prizes = getPrizes(OriginalGraph)
            num_nodes = nx.number_of_nodes(OriginalGraph)
            Nodes.append(num_nodes)
            num_edges = nx.number_of_edges(OriginalGraph)
            Edges.append(num_edges)

            print '******** City ',city,' ********'
            print 'Number of Nodes: ',num_nodes        
            print 'Number of Edges: ',num_edges  
            print
            if k > num_nodes:
                
                print 'Too few nodes!'
                IndykSol.append(None); TimesIndyk.append(None); IndykScore.append(None)
                GreedySol.append(None); TimesGr.append(None); GreedyScore.append(None)
            else:     
                print 'K:',k
                print '******** Indyk ********'
                print
                start = time.time() 

                (Graph,end_time_DP,DPTime,IndykSolutionComponent,IndykSolutionPath,\
                IndykEdgeComponent,IndykEdgePath,IndykObjValComponent,IndykObjValPath,\
                end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath,\
                tree_penalty_indyk,tree_cost_indyk,end_time_ktrees,end_time_find_tour_total,end_time_find_tour_pc)\
                                =   runIndyk(Graph,lamda,k,directory_plots,city,intersections_file,"strong")
                
                file_csv = 'sfflickr_tree_kTrees.geojson'
                create_csv(IndykSolutionComponent,file_csv,intersection_coordinates,Graph,False)
                kTreeTimes.append(end_time_ktrees)
                FindTourTimesT.append(end_time_find_tour_total)
                FindTourTimesPC.append(end_time_find_tour_pc)
                end_time_find_ktours = end_time_ktrees + end_time_find_tour_pc
                kToursTimes.append(end_time_find_ktours)
                
                end = time.time() - start
                # print 'Trees Penalty of kTrees:',tree_penalty_indyk
                VTIndyk.append(tree_penalty_indyk)
                # print 'Trees Cost of kTrees:',tree_cost_indyk
                CTIndyk.append(tree_cost_indyk)
                tree_nodes_indyk = [item for sublist in IndykSolutionComponent for item in sublist]
                path_nodes_indyk = [item for sublist in IndykSolutionPath for item in sublist]
                trees_path_penalty_indyk = computeDiffTreePath(OrigGraph,tree_nodes_indyk,path_nodes_indyk)
                # print 'Trees\Path Penalty of kTrees:',trees_path_penalty_indyk
                VPIndyk.append(trees_path_penalty_indyk)
                objective_value_Indyk,penalty,cost=\
                    computeObjectiveValue(OrigGraph,IndykSolutionComponent,IndykEdgeComponent,False)
                print 'Objective Value Indyk:',objective_value_Indyk
                # print
                # print 'Entering kPath objective value path'
                objective_value_IndykPath,penalty,cost = computeObjectiveValue(OrigGraph,IndykSolutionPath,IndykEdgePath,False)
                CPIndyk.append(cost)
                deb_sum = tree_penalty_indyk+trees_path_penalty_indyk+cost
                DebugSumIndyk.append(deb_sum) 
                # print 'Objective Value Indyk Path:',objective_value_IndykPath
                # print
                # print 'Entering Indyk objective value tour'
                objective_value_IndykTour,penalty,cost=\
                    computeObjectiveValue(OrigGraph,TourSolutionPath,TourEdgePathsTotal,False)
                # print 'Objective Value Indyk Tour:',objective_value_IndykTour
                # print
                # print 'Objective Value for k=10 and lamda',lamda,'is',objective_value_Indyk
                InTopObjVal.append(objective_value_Indyk)

                InTopObjValTour.append(objective_value_IndykTour)

                InTopObjValPath.append(objective_value_IndykPath)

                InNodesComponent.append(IndykSolutionComponent); 
                # print 'Indyk Solution Nodes Component:',IndykSolutionComponent
                # print
                # print 'Appending path to Indyk:',IndykSolutionPath
                InNodesPath.append(IndykSolutionPath);
                # print 'Indyk Solution Nodes Path:',IndykSolutionPath
                # print
                InEdgesComponent.append(IndykEdgeComponent);
                InEdgesPath.append(IndykEdgePath);
                InTime.append(end_time_DP);
                InDPTime.append(DPTime);
                InObjValComponent.append(IndykObjValComponent);
                InObjValPath.append(IndykObjValPath)

                ApNodesPath.append(TourSolutionPath);
                ApEdgesPath.append(TourEdgePath);
                ApEdgesTotalPath.append(TourEdgePathsTotal);
                ApTime.append(end_time_AP);
                InAPTime.append(TourTime);
                ApObjValPath.append(TourObjValPath)
                
                # print '***********************'
                # print
                # print '******** Greedy ********'
                # print
                # Graph = nx.Graph(OrigGraph)
                # start = time.time()

                # (Graph,end_time_DP,DPTime,GreedySolutionComponent,IndykSolutionPath,\
                # GreedyEdgeComponent,IndykEdgePath,GreedyObjValComponent,IndykObjValPath,\
                # end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath)\
                #                 = runGreedy(Graph,lamda,k,directory_plots,city,intersections_file)
                # end = time.time() - start

                # # print 'Time to run Greedy completely is:',end 
                # objective_value_Indyk_Gr,penalty,cost = \
                #     computeObjectiveValue(OrigGraph,GreedySolutionComponent,GreedyEdgeComponent,False)
                # print 'Objective Value Greedy:',objective_value_Indyk_Gr
                # # print
                # # print 'Entering Greedy+FindPath objective value path'
                # objective_value_Indyk_Gr_Path,penalty,cost = \
                #     computeObjectiveValue(OrigGraph,IndykSolutionPath,IndykEdgePath,False)
                # # print 'Objective Value Greedy Path:',objective_value_Indyk_Gr_Path
                # # print
                # # print 'Entering Greedy objective value tour'
                # objective_value_Indyk_Gr_Tour,penalty,cost = \
                #     computeObjectiveValue(OrigGraph,TourSolutionPath,TourEdgePathsTotal,False)
                # # print 'Objective Value Greedy Tour:',objective_value_Indyk_Gr_Tour
                # # print
                # GrTopObjVal.append(objective_value_Indyk_Gr)
                # # print 'Appending objective value indyk tour:',objective_value_Indyk_Gr_Tour
                # GrTopObjValTour.append(objective_value_Indyk_Gr_Tour)
                # # print 'Has value:',GrTopObjValTour
                # GrTopObjValPath.append(objective_value_Indyk_Gr_Path)

                # GrNodesComponent.append(GreedySolutionComponent); 
                # GrNodesPath.append(IndykSolutionPath);
                # GrEdgesComponent.append(GreedyEdgeComponent);
                # GrEdgesPath.append(IndykEdgePath);
                # GrTime.append(end_time_DP);
                # GrDPTime.append(DPTime);
                # GrObjValComponent.append(GreedyObjValComponent);
                # GrObjValPath.append(IndykObjValPath)

                # GrApNodesPath.append(TourSolutionPath);
                # GrApEdgesPath.append(TourEdgePath);
                # GrApEdgesTotalPath.append(TourEdgePathsTotal);
                # GrApTime.append(end_time_AP);
                # GrInAPTime.append(TourTime);
                # GrApObjValPath.append(TourObjValPath) 
                
                # print '******** 1 Tree + GreedyPaths ********'
                # Graph = nx.Graph(OriginalGraph)
                # OrigGraph = applyLamda(Graph,lamda)
                # (GRPSolutionPath, GRPSolutionEdges, objective_value) = \
                #     ptp.PCSP(Graph,OrigGraph,lamda,k,intersection_coordinates)
             
                # # print 'Outputted objective value from function:',objective_value
                # objective_value_GRP_Path,penalty,cost = \
                #     computeObjectiveValue(OrigGraph,GRPSolutionPath,GRPSolutionEdges,False)
                # print 'Objective Value 1 Tree + GreedyPaths:',objective_value_GRP_Path
                # # print 'Length Paths 1 Tree + GreedyPaths:',len(GRPSolutionPath)
                # # print 'Length Paths 1 Tree + GreedyPaths:',len(GRPSolutionPath)
                # print
                # print 
                # print '*'*50
                # GRPObjValPath.append(objective_value_GRP_Path)
                # GRPNodesPath.append(GRPSolutionPath);
                # GRPEdgesPath.append(GRPSolutionEdges);
                # # break
                # print '***********************'
                # print
                # print '******** k Trees + GreedyPaths with DP ********'
                # Graph = nx.Graph(OriginalGraph)
                # OrigGraph = applyLamda(Graph,lamda)
                # (GRDPSolutionPath,GRDPSolutionEdges,opt_score) = \
                #     ptp.PCSPGreedyDP(Graph,OrigGraph,lamda,k,intersection_coordinates)
                # # break
                # # print 'Outputted objective value from function:',objective_value
                # objective_value_GRDP_Path,penalty,cost = \
                #     computeObjectiveValue(OrigGraph,GRDPSolutionPath,GRDPSolutionEdges,False)
                # print 'Objective Value k Trees + GreedyPaths with DP:',objective_value_GRDP_Path
                # # print 'Length Paths k Trees + GreedyPaths with DP:',len(GRDPSolutionPath)
                # # print
                # # break
                # GRDPObjValPath.append(objective_value_GRDP_Path)
                # GRDPNodesPath.append(GRDPSolutionPath);
                # GRDPEdgesPath.append(GRDPSolutionEdges);
                
                # print '***********************'
                # print
                print '******** 1 Tree + k Paths ********'
                # node cur path for node: 152414766
                # node cur path for node: 4344959430
                # node cur path for node: 152539631
                # L Node is: 65317795
                Graph = nx.Graph(OriginalGraph)
                LGraph = applyLamda(Graph,lamda)
                
                (final_paths,final_edges, objective_value_kPaths, TreeGraph) = \
                    kBinarizedPCSP(LGraph,Graph,lamda,k,intersection_coordinates,"strong")
                
                file_csv = 'sfflickr_path_1Dp.geojson'
                create_csv(final_paths,file_csv,intersection_coordinates,Graph,True)
                
                objective_value_kPathDP,penalty,cost = \
                    computeObjectiveValue(LGraph,final_paths,final_edges,False)
                
                kDPObjValPath.append(objective_value_kPathDP)
                print 'Objective Value 1 Tree + k Paths:',objective_value_kPathDP
                print 'Length Paths 1 Tree + k Paths:',len(final_paths)
                # print 'Paths 1 Tree + k Paths',final_paths
                # print
                kDPNodesPath.append(final_paths);
                kDPEdgesPath.append(final_edges);
                # break
                print '***********************'
                print
                print '******** k Tree + k Paths with DP ********'
                Graph = nx.Graph(OriginalGraph)
                # print 'lalalalala',Graph.nodes(data=True)
                LGraph = applyLamda(Graph,lamda)
                (kBinDPSolutionPath,kBinDPSolutionEdges,opt_score,end_time_split_budget,end_time_find_binarize_pc,\
            end_time_find_paths_on_tree_pc,end_time_retrieve_path_pc,end_time_find_binarize_t,end_time_find_paths_on_tree_t,\
            end_time_retrieve_path_t,end_time_find_paths) = \
                    ptp.PCSP_kDP(Graph,LGraph,lamda,k,intersection_coordinates,file_plot_path,file_plot_path_img,directory)
                
                # plotPath(kBinDPSolutionPath, file_plot_path, file_plot_path_img, intersection_coordinates, node_prizes)
                file_csv = 'sfflickr_path_kDp.geojson'
                create_csv(kBinDPSolutionPath,file_csv,intersection_coordinates,Graph,True)
                
                BinarizationTimesT.append(end_time_find_binarize_t)
                BinarizationTimesPC.append(end_time_find_binarize_pc)
                SplitBudgetTimesPC.append(end_time_split_budget)
                FindPathsOnTreeTimesT.append(end_time_find_paths_on_tree_t)
                FindPathsOnTreeTimesPC.append(end_time_find_paths_on_tree_pc)
                RetrievePathTimesT.append(end_time_retrieve_path_t)
                RetrievePathTimesPC.append(end_time_retrieve_path_pc)
                FindPathsTimes.append(end_time_find_paths)
                
                objective_value_kBinDP_Path,penalty,cost = \
                    computeObjectiveValue(LGraph,kBinDPSolutionPath,kBinDPSolutionEdges,False)
                print 'Objective Value k Tree + k Paths with DP:',objective_value_kBinDP_Path
                print 'Length Paths k Tree + k Paths with DP:',len(kBinDPSolutionPath) 
                kBestDPObjValPath.append(objective_value_kBinDP_Path)
                kBestDPNodesPath.append(kBinDPSolutionPath);
                kBestDPEdgesPath.append(kBinDPSolutionEdges);
                counter+=1 
                     
    
        # break
    # print
    # print
    print '******** DBSCAN ********'
    print
    c_n = "SFFlickrDBSCAN.csv"
    DBSCAN_K_DF = pd.read_csv(c_n)
    dict_K = DBSCAN_K_DF.to_dict()
    print 'Dict_K:',dict_K
    # print dict_K
    # file_out ="DBSCAN_k"+str(K)+".html"
    FileGraph = nx.read_gml(graph_file)
    # BiggestComponent = findCompAboveThreshold(OriginalGraph,city,intersection_coordinates)
    BiggestComponent = findBiggestComponent(FileGraph,city,intersection_coordinates)
    print 'Length number of components:',len(BiggestComponent)
    OriginalGraph = nx.Graph(BiggestComponent)
     
    start = time.time()
    solution_components = []; solution_componentsGMM = []
    edge_components = []; edge_componentsGMM = []
    for k in ks:
        Graph = nx.Graph(OriginalGraph)
        OrigGraph = applyLamda(OriginalGraph,1)
        print 'K is:',k-1
        min_samples = dict_K['min_neighbors'][k-1]
        radius = dict_K['radius'][k-1]
        # print 'Min samples:',min_samples,'radius:',radius
        (solution_component,edge_component,trees_penalty,trees_cost,lats,lons) = \
             dc.runDBSCAN(Graph,OrigGraph,intersections_file,1,min_samples,radius,k)
        solution_components.append(solution_component)
        edge_components.append(edge_component)
     
    # print 'Length nodes DBSCAN Solution:',len(solution_components[0])
    # print 'Solution Component:',solution_components
    # print 'Edge Component:',edge_components
    for lamda in lamdas:
        for i, (solution_component,edge_component) in enumerate(zip(solution_components,edge_components)):      
            OriginalGraph = nx.read_gml(graph_file)      
            OrigGraph = applyLamda(OriginalGraph,lamda)
            Graph = nx.Graph(OrigGraph)
            # print 'Running for DBSCAN solution:',ks[i]
            # print
            k = ks[i]
            (Graph,end_time_DP,DPTime,DBSCANSolutionComponent,DPSolutionPath,\
            DBSCANEdgeComponent,DPEdgePath,DBSCANObjValComponent,DPObjValPath,\
            end_time_AP,TourTime,TourSolutionPath,TourEdgePathsTotal,TourEdgePath,TourObjValPath,tree_nodes,\
            tree_penalties,tree_costs,kDPSolutionPath,kDPSolutionEdges,kopt_score ) = \
                        runDensityClustering(Graph,intersections_file,solution_component,edge_component,\
                               lats,lons,directory_plots,city,k,lamda,False,'DBSCAN')
                 
            file_csv = 'sfflickr_tree_dbscan.geojson'
            create_csv(DBSCANSolutionComponent,file_csv,intersection_coordinates,Graph,False)
            VTDBSCAN.append(tree_penalty)
            CTDBSCAN.append(tree_cost)
            # print 'Trees Penalty of DBSCAN:',tree_penalty
            # print 'Trees Cost of DBSCAN:',tree_cost
            path_nodes = [item for sublist in IndykSolutionPath for item in sublist]
            trees_path_penalty = computeDiffTreePath(OrigGraph,tree_nodes,path_nodes)
            # print 'Trees\Path Penalty of DBSCAN:',trees_path_penalty
            # print 'Path Cost of DBSCAN:',trees_cost
            objective_value_DBSCAN,penalty,cost = \
                computeObjectiveValue(OrigGraph,DBSolutionComponent,DBEdgeComponent,False)
            print 'Objective Value DBSCAN:',objective_value_DBSCAN
            # print 'Objective Value DBSCAN Component:',objective_value_DBSCAN
            # print
            # print 'Entering DBSCAN+FindPath objective value path'
            objective_value_DBSCAN_Path,penalty,cost = \
                computeObjectiveValue(OrigGraph,IndykSolutionPath,IndykEdgePath,False)
            VPDBSCAN.append(trees_path_penalty)
            CPDBSCAN.append(cost)
            # print 'Objective Value DBSCAN Path:',objective_value_DBSCAN_Path
            # print
            # print 'Entering DBSCAN+FindTour objective value tour'
            objective_value_DBSCAN_Tour,penalty,cost = \
                computeObjectiveValue(OrigGraph,TourSolutionPath,TourEdgePathsTotal,False)
            # print 'Objective Value DBSCAN Tour:',objective_value_DBSCAN_Tour
            # print          
             
             
            DBSCANTopObjVal.append(objective_value_DBSCAN)
            DBSCANTopObjValTour.append(objective_value_DBSCAN_Tour)
            # print 'Has value:',GrTopObjValTour
            DBSCANTopObjValPath.append(objective_value_DBSCAN_Path)
 
            DBSCANNodesComponent.append(DBSolutionComponent); 
            DBSCANNodesPath.append(IndykSolutionPath);
            DBSCANEdgesComponent.append(DBEdgeComponent);
            DBSCANEdgesPath.append(IndykEdgePath);
            DBSCANTime.append(end_time_DP);
            DBSCANDPTime.append(DPTime);
            DBSCANObjValComponent.append(DBObjValComponent);
            DBSCANObjValPath.append(IndykObjValPath)
 
            DBSCANApNodesPath.append(TourSolutionPath);
            DBSCANApEdgesPath.append(TourEdgePath);
            DBSCANApEdgesTotalPath.append(TourEdgePathsTotal);
            DBSCANApTime.append(end_time_AP);
            DBSCANInAPTime.append(TourTime);
            DBSCANApObjValPath.append(TourObjValPath)
            end = time.time() - start
             
             
    # for i,comp in enumerate(VTDBSCAN):
    #     deb_sum = VTDBSCAN[i]+VPDBSCAN[i]+CPDBSCAN[i]
    #     DebugSumDBSCAN.append(deb_sum) 
            # print 'Objective Value DBSCAN Component:',objective_value_DBSCAN
            # print
            # print 'Entering DBSCAN+FindPath objective value path'
    #         objective_value_DBSCAN_Path,penalty,cost = \
    #             computeObjectiveValue(OrigGraph,IndykSolutionPath,IndykEdgePath,False)
    #         VPDBSCAN.append(trees_path_penalty)
    #         CPDBSCAN.append(cost)
    #         print '1 Objective Value DBSCAN Path:',objective_value_DBSCAN_Path
    #         print
    #         # print 'Entering DBSCAN+FindTour objective value tour'
    #         objective_value_DBSCAN_kPath,kpenalty,kcost = \
    #             computeObjectiveValue(OrigGraph,kDPSolutionPath,kDPSolutionEdges,False)
    #         print '2 Objective Value DBSCAN Path:',objective_value_DBSCAN_kPath
    #         print    
    #         objective_value_DBSCAN_Tour,penalty,cost = \
    #             computeObjectiveValue(OrigGraph,TourSolutionPath,TourEdgePathsTotal,False)
    #         # print 'Objective Value DBSCAN Tour:',objective_value_DBSCAN_Tour
    #         # print          
            
            
    #         DBSCANTopObjVal.append(objective_value_DBSCAN)
    #         DBSCANTopObjValTour.append(objective_value_DBSCAN_Tour)
    #         # print 'Has value:',GrTopObjValTour
    #         DBSCANTopObjValPath.append(objective_value_DBSCAN_Path)
    #         kDBSCANTopObjValPath.append(objective_value_DBSCAN_kPath)
            
    #         DBSCANNodesComponent.append(DBSolutionComponent); 
            
    #         DBSCANNodesPath.append(IndykSolutionPath);
    #         kDBSCANNodesPath.append(kDPSolutionPath)
            
    #         DBSCANEdgesComponent.append(DBEdgeComponent);
    #         DBSCANEdgesPath.append(IndykEdgePath);
    #         kDBSCANEdgesPath.append(kDPSolutionEdges);
            
    #         DBSCANTime.append(end_time_DP);
    #         DBSCANDPTime.append(DPTime);
    #         DBSCANObjValComponent.append(DBObjValComponent);
    #         DBSCANObjValPath.append(IndykObjValPath)

    #         DBSCANApNodesPath.append(TourSolutionPath);
    #         DBSCANApEdgesPath.append(TourEdgePath);
    #         DBSCANApEdgesTotalPath.append(TourEdgePathsTotal);
    #         DBSCANApTime.append(end_time_AP);
    #         DBSCANInAPTime.append(TourTime);
    #         DBSCANApObjValPath.append(TourObjValPath)
    #         end = time.time() - start
            
               
    # c_n = directory+"/Measurements/Graph"+str(city)+"_MeasurementsK.csv"
    # c_n_deb = directory+"/Measurements/Graph"+str(city)+"_DebugMeasurementsK.csv"
    # c_n_times = directory+"/Measurements/Graph"+str(city)+"_Times.csv"
    # DF_Experiment.to_csv(c_n, index=False)
    # DF_Debug.to_csv(c_n_deb, index=False)
    # DF_Times.to_csv(c_n_times,index=False)
    
if __name__ == '__main__':
    main()