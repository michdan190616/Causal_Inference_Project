import math
from scipy import stats
from scipy.stats import norm
import numpy as np
from scipy.io import loadmat
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def ci_test(D, X, Y, Z, alpha):
    n = D.shape[0]
    if len(Z) == 0:
        r = np.corrcoef(D[:, [X, Y]].T)[0][1]
    else:
        sub_index = [X, Y]
        sub_index.extend(Z)
        sub_corr = np.corrcoef(D[:, sub_index].T)
        # inverse matrix
        try:
            PM = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            PM = np.linalg.pinv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

    # Fisher’s z-transform
    res = math.sqrt(n - len(Z) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p_value = 2 * (1 - stats.norm.cdf(abs(res)))

    return p_value >= alpha

#############---------------#############
#############---------------#############

#returns true d-separation between n1, n2 given Z 
def d_sep(G, n1, n2, Z):
    return nx.d_separated(G, n1, n2, Z)

#############---------------#############
#############---------------#############

#takes a list and makes all the possible combinations of any lenghth of the elements of the list

def get_combinations(lst):
    all_combinations = []

    for r in range(1, len(lst) + 1):
        combinations_r = combinations(lst, r)
        all_combinations.extend(list(combinations_r))

    for k in range(len(all_combinations)):
        all_combinations[k] = list( all_combinations[k])
        
    return all_combinations

#############---------------#############
#############---------------#############

def tune_alpha(nodes, lis_alpha, G, data):
    j = 0
    TP = np.zeros(len(lis_alpha))
    FP = np.zeros(len(lis_alpha))
    FN = np.zeros(len(lis_alpha))

    for n1 in nodes:
        nodes.remove(n1)
        for n2 in nodes:
            
            nodes.remove(n2)
            sets = get_combinations(nodes)+[()]

            #checks the accuracy of the model getting all the possible combinations of Z
            for z in sets:
                a = d_sep(G, {n1}, {n2}, set(z))
                j = 0
                for alpha in lis_alpha : 
                    b = ci_test(data, n1, n2, list(z), alpha)
                    TP[j] += (a == 1)*(b == 1) #true positive
                    FP[j] += (a == 0)*(b == 1) #false positive
                    FN[j] += (a == 1)*(b == 0) #false negative
                    j += 1

            nodes.append(n2)
        nodes.append(n1)

    #compute of the relevant metrics of our model
    precision = np.array(TP)/np.array(TP+FP)
    recall = np.array(TP)/np.array(TP+FN)
    F1 = 2*precision*recall/(precision+recall) #most relevant metric

    alpha = lis_alpha[F1 == F1.max()].mean() #if more than one alpha has the higherst F1, it takes the average

    return alpha, F1, precision, recall

#############---------------#############
#############---------------#############

#grow phase
def grow(data, n1, alpha):

    M = []
    
    nodes = list(np.arange(data.shape[1]))
    nodes.remove(n1)

    while(True):

        M_test = M.copy()

        for j in range(len(nodes)):

            # adds elements to the temporary markov bound of n1
            if( not(ci_test(data, n1, nodes[j], M, alpha))):
                M.append(nodes[j])
                nodes.remove(nodes[j])
                break
        
        #if M is not changed through the entire cycle it means that we found the Grown markov boundary
        if M_test == M:
            break
    
    return M

#############---------------#############

#shrink phase
def shrink(data, n1, alpha, M ):


    while(True):

        M1 = M.copy()

        for j in range(len(M)):

            M2 = M.copy()
            M2.remove(M2[j])

            #removes elements from the temporary Markov bound of n1
            if( ci_test(data, n1, M[j], M2, alpha)):
                M.remove(M[j])
                break

        #if M is not changed through the entire cycle it means that we found the shrunk markov boundary
        if M1 == M:
            break
    
    return M

#############---------------#############

#returns all the markov boundary of the graph and the moralized graph
def GS_step1(data, alpha):

    M = []

    #finds grown markov bounds for every node
    for j in range(data.shape[1]):

        M_temp1 = grow(data, j, alpha)
        M_temp2 = shrink(data, j, alpha, M_temp1)

        M.append(M_temp2)
    
    #finds the markov boundary for every node
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if j in M[i] and i not in M[j]:
                M[i].remove(j)

    G = nx.empty_graph(data.shape[1],create_using=nx.Graph())

    #from the data  about the markov boundaries builds the graph
    for j in range(data.shape[1]):
        for i in M[j]:

            G.add_edge(j,i)

    return G,M

#############---------------#############
#############---------------#############

#undirected graph
def skeleton(G, data, alpha, M):

    nodes = list(np.arange(data.shape[1]))
    separation=dict()
    neighbors = []

    for a in nodes:
        for b in nodes:
            if b!=a:
                separation[(a,b)]=None

    for j in nodes:
        neighbors = list(G.neighbors(j)).copy()

        for i in neighbors:

            #smaller between Mb(x)\y and Mb(y)\x
            m1_temp = M[j].copy()
            m2_temp = M[i].copy()            
            
            if i in M[j]:
                m1_temp.remove(i)
            
            if j in M[i]:
                m2_temp.remove(j)
            m3_temp = m1_temp.copy()
            
            if len(m1_temp)>len(m2_temp):

                m3_temp = m2_temp.copy()
                
            S = get_combinations(m3_temp)+[[]]

            temp = True

            #check if i and j are neighbours, testing the d-separation give the smaller between Mb(x)\y and Mb(y)\x
            for k in S:

                if (ci_test(data, i, j, k, alpha)):
                    temp = False
                    separation[(j,i)] = k
                    break
            
            #if it finds that at least one element is not independent, it removes the edge
            if not(temp) and G.has_edge(i,j):
                G.remove_edge(i,j)
    return G, separation

#############---------------#############

#builds the v-structs
def v_struct(A, G, separation):
    for j in list(np.argwhere(A.sum(axis=1) >= 2).reshape(-1)):
        neighbors = list(np.argwhere(A[j, :] == 1).reshape(-1))
        for i in neighbors:
            temp_neighbors = neighbors.copy()
            temp_neighbors.remove(i)
            #checks if there is no edge between i and j, then it checks if i and j are d-separated given an arbitrary separating set
            #then it checks if in j is in that separating set
            for k in temp_neighbors:
                if A[i, k] == 0 and A[k, i] == 0 and separation[(i, k)] is not None and j not in separation[(i, k)] and (A[j, i] == 1 or A[j, k] == 1):
                    if A[i, j] == 1 and A[j, i] == 1:
                        A[j, i] = 0
                        G.remove_edge(j, i)                       
                    if A[k, j] == 1 and A[k, j] == 1:
                        A[j, k] = 0
                        G.remove_edge(j, k)
                        

    return A, G

#############---------------#############

#returns the undirected graph with the v structures
def GS_step2(data, alpha, G, M):

    G,separation=skeleton(G, data, alpha, M)
    A=(nx.adjacency_matrix(G)).toarray()
    G=nx.from_numpy_array(A,create_using=nx.DiGraph())
    A,G=v_struct(A,G,separation)

    return G, A

#############---------------#############
#############---------------#############

#function that implements the meek algorithm to orient the graph
def meek(G, A):
    # Rule 1
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                if A[i, j] == 1 and A[j, i] == 0 and A[j, k] == 1 and A[k, j] == 1 and A[i, k] == 0 and A[k, i] == 0:
                    G.remove_edge(k, j)
                    A[k, j] = 0

    # Rule 2
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                if A[i, j] == 1 and A[j, i] == 0 and A[j, k] == 1 and A[k, j] == 0 and A[i, k] == 1 and A[k, i] == 1:
                    G.remove_edge(k, i)
                    A[k, i] = 0

    # Rule 3
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                for q in range(A.shape[0]):
                    if A[i, j] == 1 and A[j, i] == 1 and A[i, k] == 1 and A[k, i] == 1 and A[i, q] == 1 and A[q, i] == 1 and A[j, q] == 1 and A[q, j] == 0 and A[k, q] == 1 and A[q, k] == 0 and A[j, k] == 0 and A[k, j] == 0:
                        G.remove_edge(q, i)
                        A[q, i] = 0

    # Rule 4
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                for q in range(A.shape[0]):
                    if A[i, j] == 1 and A[j, i] == 1 and A[i, k] == 1 and A[k, i] == 1 and A[i, q] == 1 and A[q, i] == 1 and A[q, j] == 1 and A[j, q] == 0 and A[k, q] == 1 and A[q, k] == 0 and A[j, k] == 0 and A[k, j] == 0:
                        G.remove_edge(j, i)
                        A[j, i] = 0

    return G

#############---------------#############

#returns the oriented graph
def GS_step3(G, A):

    G =  meek(G,A)

    return G

#############---------------#############
#############---------------#############

