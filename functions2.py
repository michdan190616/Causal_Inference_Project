import itertools
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pulp import *
import random
import seaborn as sns
sns.set(rc={'figure.figsize':(15,8)})


def Hhull(S, Gd, Gu):
    # This version of the Hhull function only works for sets S which are c-components
    # Initialize F initial value as the entire set of nodes of the directed Graph (Gd)
    F = set(Gd.nodes())

    while True:
        # Compute the maximal c-component in the subgraph G[F] that contains S
        F1 = maximal_c_component(Gu, F, S)
        # Define F2 as the ancestors of S in F1
        F2 = ancestors(Gd.subgraph(F1), S)
        
        # Exit condition: we need to return F2 = F 
        if F2 != F:
            F = F2
        else:
            return F

def maximal_c_component(Gu, F, S):
    # Function that returns the maximal c-component containing S in the subgraph G[F]
    GuF = Gu.subgraph(F)

    # Create a list of all the c-components in G[F]
    valid_c_components = [set(component) if len(component.intersection(S)) >0 else set() 
                          for component in nx.connected_components(GuF)]
    # Choose the c-component with the bigger number of elements in common with S
    maximal_component = max(valid_c_components, key=lambda component: len(component))
    return maximal_component


def ancestors(Gd, S):
    # Returns the ancestors of S in the directed graph Gd
    # If we get a list of nodes we convert it to Set type
    if type(S) == list:
        ancestors = set(S)
    
    ancestors = S.copy()
    
    # Add ancestors of each node in S to the output set
    for vertex in S:
        ancestors.update(nx.ancestors(Gd, vertex))
    return ancestors


def set_weight(set,weights):
    # Function useful to compute the total weight (cost) of a set given a dictionary of weights
    return sum(weights[elem] for elem in set)


def min_hitting_set(sets_list, weights):
    # Find the set with the minimum cost containing elements from all the sets in a given list
    if len(sets_list)>1:
        # Create list containing all possible hitting sets
        hitting_list = list(itertools.product(*[list(sets) for sets in sets_list]))
        hitting_sets = [set(lis) for lis in hitting_list]

        # Find the minimum weight hitting set among all subsets using set_weight function
        min_hitting_set = min(hitting_sets,key=lambda x: set_weight(x,weights))
        return min_hitting_set

    else:
        # For a single set, return the element with lower weight
        return {str(min(sets_list[0],key=lambda x: weights[x]))}
    

def MinCostIntervention(S, Gd, Gu, C):
    # Function that returns the minimum cost intervention set
    # This version only allows for a set S being a c-component
    F = []

    H = Hhull(S, Gd, Gu)
    if H == S:
        return set()

    while True:
        while True:
            #Compute argmin of cost over the set H\S
            a = min(H - S, key=lambda v: C[v])
            #Compute maximal hedge for Q[S] over the subgraph G[H\S]
            H_minus_a = Hhull(S, Gd.subgraph(H-{a}), Gu.subgraph(H-{a}))
            if H_minus_a == S:
                #Add H to the list F and exit the inner loop
                F.append(H)
                break
            else:
                H = H_minus_a

        # Remove the S elements from the F sets
        sets_minus_S = [element - S for element in F]
        # Compute minimum hitting set for F\S
        hitting_set = min_hitting_set(sets_minus_S,C)
        # All nodes except for the hitting set
        V_minus_hitting_set = Gd.nodes - hitting_set
        # Maximum hedge for Q[S] over the subgraph containing all nodes but the hitting set
        H_minus_hitting_set = Hhull(S, Gd.subgraph(V_minus_hitting_set), Gu.subgraph(V_minus_hitting_set))
        if H_minus_hitting_set == S:
            return hitting_set
        else:
            H = H_minus_hitting_set


def min_nodes_cut(gu,source_set,S,weights):
    # Algorithm that solves a minimum vertex cut over an undirected graph with weighted nodes
    # The approach is to transform it into a directed graph with weighted edges and solve using
    # Standard minimum cut routine

    # Create directed graph
    weighted_edges_graph = nx.DiGraph()

    # Add source node and link it to the vertices in the source set
    # Put infinite weight to this edges: we only want to intervene on single nodes
    for node in source_set:
        weighted_edges_graph.add_edge('source',str(node)+'_in',capacity = np.inf)
    
    # Every node in the S set is 'splitted' into an input and an output node, with an infinitely
    # weighted edge linking the two: we cannot intervene on the S set
    # S nodes have an edge linking them to the target node, with infinite weight
    for node in S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity = np.inf)
        weighted_edges_graph.add_edge(str(node)+'_out','target',capacity = np.inf)

        # As before, put infinite weight also on edges between different nodes
        for neighbor in gu.adj[node]:
            weighted_edges_graph.add_edge(str(node)+'_out',str(neighbor)+'_in',capacity = np.inf)
    
    # For the remaining nodes, we add the cost of the node's intervention to the weight linking
    # input and output nodes: cutting the edge is equivalent to intervening on the node
    for node in gu.nodes - S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity=weights[node])
        # Edges to different nodes have infinite weight
        for neighbor in gu.adj[node]:
            weighted_edges_graph.add_edge(str(node)+'_out',str(neighbor)+'_in',capacity = np.inf)

    # Execute a minimum cut algorithm on the new graph to obtain the cost and the two cut sets
    cost, cut_sets = nx.minimum_cut(weighted_edges_graph,'source','target')

    setA , setB = cut_sets


    # Create a set with the nodes having input and output nodes in two different partitions: these are the minimum cut nodes
    cut_set = set()
    for node in gu.nodes - S:
        if (str(node)+'_in' in setA and str(node)+'_out' in setB) or (str(node)+'_in' in setB and str(node)+'_out' in setA):
            cut_set.update({node})

    return (cost,cut_set)


def Heuristic(S, Gd, Gu, C):
    # Euristic Algorithm using the minimum cut approach

    # Find set of all parents of S
    parents = set()
    for s in S:
        parents_s = set(Gd.predecessors(s))
        parents.update(parents_s)  

    # Find maximum hedge for Q[S] in the given graph
    H = Hhull(S=S,Gd=Gd,Gu=Gu)
    # Define source set H1 for the minimum cut problem as the intersection of the hedge and the parent sets
    H1 = H & set(parents-S)

    # If the source set is empty, no need for intervention
    if H1 == set():
        return set()

    # Minimum vertex cut with S as target set and H1 as source set
    cost,A = min_nodes_cut(Gu.subgraph(H),source_set=H1,S=S,weights = C)

    return A


def MinCostInterventionMult(S, Gd, Gu, C):
    # N_nodes_nodes_nodes_nodesew version of the minimum cost intervention algorithm allowing for multiple maximal c-components

    c_components = []
    for s in S:
        # Use function defined before to find maximal c-components containing at least one element from S
        max_c_component = maximal_c_component(Gu, Gu.nodes, s)
        # Add the component to the list only if it is not included yet
        if max_c_component not in c_components:
            c_components.append(max_c_component)

    # Call the previous version of the algorithm on each S partition an do the union of the intervention sets
    intervention_set = set()
    for component in c_components:
        S_partition = component.intersection(S)
        intervention_set.update(MinCostIntervention(S_partition,Gd,Gu,C))
    
    return intervention_set


def HeuristicMult(S, Gd, Gu, C):
# N_nodesew version of the heuristic algorithm allowing for multiple maximal c-components
    
    # Find list of maximal c-components containing elements from S
    c_components = []
    for s in S:
        max_c_component = maximal_c_component(Gu, Gu.nodes, s)
        if max_c_component not in c_components:
            c_components.append(max_c_component)

    #Run the Heuristic algorithm for each S partition and return union of intervention sets
    intervention_set = set()
    for component in c_components:
        S_partition = component.intersection(S)
        intervention_set.update(Heuristic(S_partition,Gd,Gu,C))

    return intervention_set
# Linear Programming approach using pulp library


def hitting_set_LP(sets_list, costs):
    # Function returning approximation of optimal hitting set: 
    # by relaxing the integer constraint, we include in the hitting set all decision variables with 
    # x(i) > 1/k, where k is the maximum set size

    # Create single set from union of all sets
    sets_union = set().union(*sets_list)
    L = len(sets_list)
    if len(sets_list)>1:
        # Initialize the minimization problem
        problem = LpProblem("hitting_set",LpMinimize)
        
        # Create the decision variables x, one for each node
        # Only impose positivity constraint (not binary)
        x = LpVariable.dicts("x", sets_union, lowBound=0)
        
        # Add the objective function (total cost to minimize) to the problem:
        problem += lpSum(costs[element] * x[element] for element in sets_union)
        
        # Add constraint of at least a total weight of 1 for each set
        for set_elements in sets_list:
            problem += lpSum(x[element] for element in set_elements) >= 1
        
        # Solve the minimization problem
        problem.solve()
        
        # Add to the hitting set only the decision variables with value greater than 1/k
        k = max(len(s) for s in sets_list)

        hitting_set = {element for element in sets_union if value(x[element]) >= 1/k}
        return hitting_set

    # If we only have one set, return the element with minimum cost
    elif len(sets_list) == 1: 
        return {min(sets_list[0],key=lambda x: costs[x])}


def MinCostIntervention_LP(S, Gd, Gu, C):
    # Function that runs algorithm similar to MinCostIntervention, 
    # using the LP approach to compute hitting set 
    F = []

    H = Hhull(S, Gd, Gu)
    if H == S:
        return set()

    while True:
        while True:
            a = min(H - S, key=lambda v: C[v])
            H_minus_a = Hhull(S, Gd.subgraph(H-{a}), Gu.subgraph(H-{a}))
            if H_minus_a == S:
                F.append(H) 
                break
            else:
                H = H_minus_a

        sets_minus_S = [element - S for element in F]
        # Use Linear Programming algorithm to compute approximation of optimal hitting set
        hitting_set = hitting_set_LP(sets_minus_S,C)

        V_minus_hitting_set = Gd.nodes - hitting_set
        H_minus_hitting_set = Hhull(S, Gd.subgraph(V_minus_hitting_set), Gu.subgraph(V_minus_hitting_set))
        if H_minus_hitting_set == S:
            return hitting_set
        else:
            H = H_minus_hitting_set


class random_ADMG:
    # Create class to randomize graph creation
    def __init__(self,N_nodes,Nd_edges,Nu_edges):
        # Create a set of nodes
        nodes = [str(i) for i in range(1, N_nodes+1)]
        # Create a connected directed graph with Nd_edges
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(nodes)
        directed_edges = random.sample([(u, v) for u in nodes for v in nodes if u > v], k=Nd_edges)
        directed_graph.add_edges_from(directed_edges)

        # Create an undirected graph with Nu_edges/2 edges overlapping the directed ones, and Nu_edges/2 from new connections
        undirected_graph = nx.Graph()
        undirected_graph.add_nodes_from(nodes)
        undirected_edges = random.sample(directed_edges, k=int(Nu_edges/2))
        undirected_edges+=(random.sample([(u, v) for u in nodes for v in nodes if (u>v and ((u,v) not in directed_edges))], k=Nu_edges-int(Nu_edges/2)))
        undirected_graph.add_edges_from(undirected_edges)



        # Find the unconnected subgraphs and add edges to connect them
        unconnected_subgraphs = list(nx.strongly_connected_components(directed_graph))
        num_subgraphs = len(unconnected_subgraphs)
        if num_subgraphs > 1:
            for i in range(num_subgraphs - 1):
                source_node = max(list(unconnected_subgraphs[i])[0],list(unconnected_subgraphs[i+1])[0])
                target_node = min(list(unconnected_subgraphs[i])[0],list(unconnected_subgraphs[i+1])[0])
                directed_graph.add_edge(source_node, target_node)

        # Add random weights from 1 to 20 for each node
        weights = {node:random.random() for node in nodes}

        self.gu = undirected_graph
        self.gd = directed_graph
        self.nodes = nodes
        self.weights = weights
    
    def S_set(self):
        # Find the maximal c-component from the undirected graph
        c_components = [set(component) for component in nx.connected_components(self.gu)]
        maximal_component = max(c_components, key=lambda component: len(component))

        # For better simulations, we return nodes with parents in the maximal component
        child_nodes = set()
        for node in maximal_component:
            parents = nx.ancestors(self.gd,node) - {node}
            if parents.intersection(maximal_component) != set():
                child_nodes.update(node)

        c_comp = [set(component) for component in nx.connected_components(self.gu.subgraph(child_nodes))]
        conn_nodes = max(c_comp, key=lambda component: len(component))

        #Return the nodes meeting the conditions above
        return conn_nodes
            


    def draw(self, title=None):
        # Create a figure and subplots
        fig, axs = plt.subplots(1, 2)

        if title:
            fig.suptitle(title, fontsize=18)

        # Draw the undirected graph
        axs[0].set_title("Undirected Graph", size=14)
        nx.draw(self.gu, pos = nx.circular_layout(self.gu),ax=axs[0], with_labels=True, node_color='lightblue')

        # Draw the directed graph
        axs[1].set_title("Directed Graph", size=14)
        nx.draw(self.gd, pos = nx.circular_layout(self.gd),ax=axs[1], with_labels=True, node_color='lightgreen')

        plt.subplots_adjust(wspace=0.3)

        plt.show()
