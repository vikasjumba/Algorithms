#http://www.codeskulptor.org/#user40_SklG1I2vt7_5.py
"""
Provided code for Application portion of Module 2
"""

# general imports
import urllib2
import random
import time
import math

# CodeSkulptor import
#import simpleplot
#import codeskulptor
#codeskulptor.set_timeout(1000)

# Desktop imports
import matplotlib.pyplot as plt
import random

class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm
    
    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that each node number
        appears in correct ratio
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order
    


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


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

"""
Implementation of assignment 2 Algorithmic thinking
"""
"""
Queue class
"""

class Queue:
    """
    A simple implementation of a FIFO queue.
    """

    def __init__(self):
        """ 
        Initialize the queue.
        """
        self._items = []

    def __len__(self):
        """
        Return the number of items in the queue.
        """
        return len(self._items)
    
    def __iter__(self):
        """
        Create an iterator for the queue.
        """
        for item in self._items:
            yield item

    def __str__(self):
        """
        Return a string representation of the queue.
        """
        return str(self._items)

    def enqueue(self, item):
        """
        Add item to the queue.
        """        
        self._items.append(item)

    def dequeue(self):
        """
        Remove and return the least recently inserted item.
        """
        return self._items.pop(0)

    def clear(self):
        """
        Remove all items from the queue.
        """
        self._items = []
        
  
def bfs_visited(ugraph, start_node):
    """
    Implements visited bfs algorithm 
    Return set of visited nodes
    """
    if ugraph.get(start_node,0) == 0:
        return set([])
    bfs_que = Queue()
    bfs_que.enqueue(start_node)   
    visited = set([start_node])
    while len(bfs_que) > 0:
        node = bfs_que.dequeue()
        for neigh in ugraph[node]:
            if neigh not in visited:
                visited.add(neigh)
                bfs_que.enqueue(neigh)

    return visited

def cc_visited(ugraph):
    """
    cc_visited implemented
    """
    rem_nodes = set(ugraph.keys())
    conn_comp = []
    while len(rem_nodes) > 0:
        node = rem_nodes.pop()
        visited = bfs_visited(ugraph,node)
        conn_comp.append(visited)
        rem_nodes = rem_nodes.difference(visited)
    return conn_comp
def largest_cc_size(ugraph):
    """
    Computes largest cc
    """
    conn_comp = cc_visited(ugraph)
    larg_cc_size = 0
    for comp in conn_comp:
        size = len(comp)
        if larg_cc_size < size:
            larg_cc_size = size    
    return larg_cc_size

def compute_resilience(ugraph, attack_order):
    """
    Return size of graph largest cc while attack propagation
    """
    node_numb = 0
    res = [largest_cc_size(ugraph)]    
    for node in attack_order:
        del ugraph[node]
        node_numb += 1 
        for key in ugraph.keys():
            neigh = ugraph.get(key)
            neigh = neigh.difference(set([node]))
            ugraph[key] = neigh
            
        res.append(largest_cc_size(ugraph))
    return res

def generate_er_graph(nodes,prob):
    res = {node: set([]) for node in range(nodes)}
    
    if nodes > 0:
        for node in range(0,nodes):
            for edge_node in range(0,nodes):
                if edge_node != node  and random.random() < prob:
                    res[node].add(edge_node)
		    res[edge_node].add(node)
            
    return res   

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
                #alrdy = res.get(edge_node,[])
                if edge_node != node: #and node not in alrdy:
                    adj_nodes.append(edge_node)
            res[node] = set(adj_nodes)
    return res    

def load_upa(m,n):
    res = make_complete_graph(m)
    upa_obj = UPATrial(m)
    for idx in range(m,n):
        neighbors = upa_obj.run_trial(m)
        res[idx] = neighbors
        for neighbor in neighbors:
	    res[neighbor].add(idx)
    return res

     
def get_avg_outdegree(graph):
    counter = 1
    tot_out_deg = 0
    for node in graph.keys():
        out_deg = len(graph.get(node))
        tot_out_deg += out_deg
        counter += 1
    return (float(tot_out_deg)/2, float(tot_out_deg)/(2*counter))            
def fast_targeted_order(ugraph):
    new_graph = copy_graph(ugraph)
    nodes = len(ugraph.keys())
    deg_sets = [set([])] * nodes
    for node, neighbors in new_graph.iteritems():
        deg = len(neighbors)
	deg_sets[deg].add(node)
    res_l = []
    #idx = 0
    for deg in range(nodes-1,-1,-1):
	while len(deg_sets[deg]) > 0:
              sel_node = deg_sets[deg].pop()
	      for neigh in new_graph[sel_node]:
        	   deg_neigh = len(new_graph.get(neigh))
		   deg_sets[deg_neigh].remove(neigh) 	
		   deg_sets[deg_neigh-1].add(neigh)

	      res_l.append(sel_node)
	      #idx += 1
	      neighbors = new_graph[sel_node]
              new_graph.pop(sel_node)
              for neighbor in neighbors:
                  new_graph[neighbor].remove(sel_node)

    return res_l

def plot_q1(is_targeted):
    prob = 0.002
    n = 1239
    m = 3
    ngraph = load_graph(NETWORK_URL)
    ergraph = generate_er_graph(n,prob)
    upgraph = load_upa(m,n)
    
    nrand = ngraph.keys()
    
    errand = ergraph.keys()


    uprand = upgraph.keys()

    if is_targeted:
       nrand = targeted_order(ngraph)
       errand = targeted_order(ergraph)
       uprand = targeted_order(upgraph)
    else:
       random.shuffle(nrand)
       random.shuffle(errand) 
       random.shuffle(uprand)
 	
    n_data = compute_resilience(ngraph, nrand)
    er_data = compute_resilience(ergraph, errand)
    up_data = compute_resilience(upgraph, uprand)
    fig = plt.figure()
    plt.plot(range(0,len(nrand)+1), n_data, '-b', label='Network')
    plt.plot(range(0,len(errand)+1), er_data, '-r', label='ER (P = 0.002)')   
    plt.plot(range(0,len(uprand)+1), up_data, '-g', label='UPA (m = 3)')
    plt.title('Graph Resiliency')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Nodes Removed')
    plt.ylabel('Size of Largest Connected Component')
    plt.show()
    if is_targeted:
    	fig.savefig('Q4.png')
    else:
	fig.savefig('Q1.png')
#import time    
def plot_q3():
    y_target = []
    y_fast_target = []
    for n in range(10, 1000, 10): 
	ugraph = load_upa(5,n)
    	st_time = time.time()
    	res = targeted_order(ugraph)    
    	y_target.append(time.time() - st_time)
    	st_time = time.time()
    	res = fast_targeted_order(ugraph)
    	y_fast_target.append(time.time() - st_time)

    fig = plt.figure()
    plt.plot(range(10,1000,10),y_target, '-b', label='Target Order')
    plt.plot(range(10,1000,10), y_fast_target, '-g', label='Fast Target Order')   
    plt.title('Running Time Comparison (Desktop Python)')
    plt.legend(loc='upper left')
    plt.xlabel('Number of Nodes (UPA, m = 5)')
    plt.ylabel('Running Time (Sec)')
    plt.show()
    fig.savefig('Q3.png')

is_targeted = True
plot_q1(is_targeted)
#plot_q3()
#plot_q4()
#ngraph = load_graph(NETWORK_URL)
#n = len(ngraph.keys())
#print n
#(edges,avg)= get_avg_outdegree(ngraph)
#print edges
#print 'Prob =', edges/(n*(n-1)/2)
#print 'M =', avg 

#upa_graph = load_upa(3,n)
#(edges,avg)= get_avg_outdegree(upa_graph)
#print 'UPA', edges

#er_g = generate_er_graph(n,0.002)
#(edges,avg)= get_avg_outdegree(er_g)
#print 'ER', edges



