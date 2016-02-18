http://www.codeskulptor.org/#user40_u78mXbLtE8_4.py
"""
Sample graphs 
"""
EX_GRAPH0 = {0:set([1,2]),
             1:set([]),
             2:set([])} 
EX_GRAPH1 = {0:set([1,4,5]),
             1:set([2,6]),
             2:set([3]),
             3:set([0]),
             4:set([1]),
             5:set([2]),
             6:set([])} 
EX_GRAPH2 = {0:set([1,4,5]),
             1:set([2,6]),
             2:set([3,7]),
             3:set([7]),
             4:set([1]),
             5:set([2]),
             6:set([]), 
             7:set([3]),
             8:set([1,2]),
             9:set([0,3,4,5,6,7])}

def make_complete_graph(num_nodes):
    """
    Takes number of nodes as input 
    Return complete graph
    """
    res = {}
    if num_nodes > 0:
        for node in range(0,num_nodes):
            adj_nodes = []
            for edge_node in range(0,num_nodes):
                if edge_node != node:
                    adj_nodes.append(edge_node)
            res[node] = set(adj_nodes)
    return res    
def compute_in_degrees(digraph):
    """
    Takes dictionary representation of graph as input 
    Return in_degree count for each node in the form of dictionary 
    """
    res = dict()
    for key in digraph:
        res[key] = 0   
    for key in digraph:
        tail_edges = digraph.get(key)                  
        for edge in tail_edges:
            res[edge] = res.get(edge,0) + 1
    return res  
def in_degree_distribution(digraph):
    """
    Takes dictionary representation of graph as input 
    Return in_degree distribution in the form of dictionary 
    """
    in_degrees  = compute_in_degrees(digraph)
    res = {}
    for val in in_degrees.values():
        res[val] = res.get(val,0) + 1
    return res

	
http://www.codeskulptor.org/#alg_dpa_trial.py
"""
Provided code for application portion of module 1

Helper class for implementing efficient version
of DPA algorithm
"""

# general imports
import random


class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm
    
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a DPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors
    




http://www.codeskulptor.org/#user40_It1T1VkGjX_10.py

"""
Provided code for Application portion of Module 1

Imports physics citation graph 
"""

# general imports
import urllib2
import user40_u78mXbLtE8_4 as Graph
import simpleplot
import math
import random
import alg_dpa_trial as dpa
# Set timeout for CodeSkulptor if necessary
import codeskulptor
codeskulptor.set_timeout(600)


###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


def get_plot_data(digraph):
    deg_dist = Graph.in_degree_distribution(digraph)
    node_num = len(digraph.keys())
    plot_lst = []
    for ind in range(1,node_num):
        y_val = deg_dist.get(ind)
        if y_val != None and y_val != 0:
            y_val = float(y_val) / node_num
            plot_lst.append([math.log(ind),math.log(y_val)])
            
    return plot_lst

def generate_er_graph(nodes,prob):
    res = {}
    if nodes > 0:
        for node in range(0,nodes):
            adj_nodes = []
            for edge_node in range(0,nodes):
                if edge_node != node and random.random() < prob:
                    adj_nodes.append(edge_node)
            res[node] = set(adj_nodes)
    return res   
def get_avg_outdegree(graph):
    counter = 1
    tot_out_deg = 0
    for node in graph.keys():
        out_deg = len(graph.get(node))
        tot_out_deg += out_deg
        counter += 1
    return float(tot_out_deg)/counter

def load_dpa(m,n):
    res = Graph.make_complete_graph(m)
    dpa_obj = dpa.DPATrial(m)
    for idx in range(m,n):
        neighbours = dpa_obj.run_trial(m)
        res[idx] = neighbours
    return res

M = 13
N = 27770
dpa_graph = load_dpa(M,N)
citation_graph = load_graph(CITATION_URL)
#m = get_avg_outdegree(citation_graph)
#print 'm = ', m

plst1 = get_plot_data(citation_graph)
plst2 = get_plot_data(dpa_graph)
nodes = 1000
plst3 = get_plot_data(generate_er_graph(nodes,0.1))

simpleplot.plot_lines("In-Degree Normalized Distribution (Citation)", 600, 600,"log_e(Degree)", "log_e(Frequency)", [plst1],True,['Citation'])

simpleplot.plot_lines("In-Degree Normalized Distribution (DPA)", 600, 600,"log_e(Degree)", "log_e(Frequency)", [plst2],True,['DPA'])

simpleplot.plot_lines("In-Degree Normalized Distribution (ER)", 600, 600,"log_e(Degree)", "log_e(Frequency)", [plst3],True,['ER'])

simpleplot.plot_lines("In-Degree Normalized Distribution Comparison (Citation vs DPA)", 600, 600,"log_e(Degree)", "log_e(Frequency)", [plst1,plst2],True,['Citation','DPA'])
    
simpleplot.plot_lines("In-Degree Normalized Distribution Comparison (Citation vs ER)", 600, 600,"log_e(Degree)", "log_e(Frequency)", [plst1,plst3],True,['Citation','ER'])
























