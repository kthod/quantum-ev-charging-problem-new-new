from qiskit import *
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time
from VQE_Class import*
from CompressedVQE_Class import*
from CompressedVQE_RPA_Class import*
from ClusterVQE_Class import*
from ClusterCompressedVQE_Class import*
import matplotlib.colors as mcolors


def get_qubomat3(eps,delta,Horizon,V,DT):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    d = np.ones(2*Horizon)
    #print(eps)
# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 0
    #print(d)
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    p = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(d),len(d)))
    nvar = len(matrix)
    
    for i in range(len(d)):
        for j in range(i,len(d)):
            if i == j:
                matrix[i][i] = 256*(V**2)*(DT**2) * ((p[i]**2)*(d[i]**2)) - 16*2*eps*V*DT*d[i]*p[i]# + 256*p[i]**2
            else:
                matrix[i][j] = 256*2*(V**2)*(DT**2) *p[i]*p[j]*d[i]*d[j]
    return matrix


def get_qubomat4(evi,evj,deltai,deltaj,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    di = np.ones(2*Horizon)
    dj = np.ones(2*Horizon)
# Then, change the elements after the first k elements to 1000
    di[2*deltai:] = 0
    dj[2*deltaj:] = 0
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(di),len(di)))
    nvar = len(matrix)
    if evi==evj:
        for i in range(0,len(di),2):
 
            matrix[i:(i+2),i:(i+2)] =  256*np.array([[(pi[i]**2)*di[i]**2,2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],[0,(pi[i+1]**2)*di[i+1]**2]]) 
    else:
        for i in range(0,len(di),2):
            matrix[i:(i+2),i:(i+2)] =2*256*np.array([[(pi[i]**2)*di[i]*dj[i],(pi[i]*pi[i+1])*(di[i]*dj[i+1])],[(pi[i]*pi[i+1])*(dj[i]*di[i+1]),(pi[i+1]**2)*di[i+1]*dj[i+1]]])           
    return matrix


def get_qubomat_energy_limit1(evi,evj,deltai,deltaj,Horizon,V,C):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    di = np.ones(2*Horizon)
    dj = np.ones(2*Horizon)
# Then, change the elements after the first k elements to 1000
    di[2*deltai:] = 0
    dj[2*deltaj:] = 0
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(di),len(di)))
    nvar = len(matrix)
    if evi==evj:
        for i in range(0,len(di),2):
 
            matrix[i:(i+2),i:(i+2)] =  np.array([[256*V**2*(pi[i]**2)*di[i]**2 - 2*C*V*16*pi[i]*di[i] ,256*V**2*2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],[0, 256*V**2*(pi[i+1]**2)*di[i+1]**2 - 2*C*V*16*pi[i+1]*di[i+1]]]) 
    else:
        for i in range(0,len(di),2):
            matrix[i:(i+2),i:(i+2)] =2*256*V**2*np.array([[(pi[i]**2)*di[i]*dj[i],(pi[i]*pi[i+1])*(di[i]*dj[i+1])],[(pi[i]*pi[i+1])*(dj[i]*di[i+1]),(pi[i+1]**2)*di[i+1]*dj[i+1]]])           
    return matrix

def get_qubomat_energy_limit2(V,C,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.

   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+3) % 3) for i in range(4*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((4*Horizon,4*Horizon))
    nvar = len(matrix)
 
    for i in range(0, 4*Horizon,4):
        matrix[i:(i+4),i:(i+4)] =  np.array([[256*V**2*(pi[i]**2)- 2*C*V*16*pi[i] ,256*V**2*2*(pi[i]*pi[i+1]),256*V**2*2*(pi[i]*pi[i+2]),256*V**2*2*(pi[i]*pi[i+3])],
                                            [0, 256*V**2*(pi[i+1]**2) - 2*C*V*16*pi[i+1], 256*V**2*2*(pi[i+1]*pi[i+2]), 256*V**2*2*(pi[i+1]*pi[i+3])], 
                                            [0, 0, 256*V**2*(pi[i+2]**2) - 2*C*V*16*pi[i+2], 256*V**2*2*(pi[i+2]*pi[i+3])],
                                            [0, 0, 0, 256*V**2*(pi[i+1]**2) - 2*C*V*16*pi[i+1]]  ]) 
    
    return matrix

def get_qubomat_energy_limit3(delta, V,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    d = np.ones(2*Horizon)
    
# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 0
    
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+3) % 4) for i in range(4*Horizon)])
    matrix = np.zeros((len(d),4*Horizon))
    nvar = len(matrix)
   

 
    for i in range(0, Horizon):
        matrix[i*2:(i+1)*2,i*4:(i+1)*4] = 2*256*V**2*np.array([[pi[i]*d[i]*pj[i], pi[i]*d[i]*pj[i+1], pi[i]*d[i]*pj[i+2], pi[i]*d[i]*pj[i+3]],
                                                                [pi[i+1]*d[i+1]*pj[i], pi[i+1]*d[i+1]*pj[i+1], pi[i+1]*d[i+1]*pj[i+2], pi[i+1]*d[i+1]*pj[i+3]]])            
   
    return matrix

def ret_schedule(solution,evs, Horizon,de):
    sched = np.zeros((evs,Horizon))
    for i in range(evs):
        sol = solution[i]
        #bitstring = np.array([int(x) for x in sol])
        for j in range(Horizon):
            if j<de[i]:
                sched[i,j] = 16*int(sol[2*j:2*j+2],2)
            else:
                sched[i,j] = 0

    
    return sched

def reshape_solution(sol,evs, Horizon):
    solution = []
    for i in range(evs):
        solution.append(sol[i*2*Horizon:(i+1)*2*Horizon])
        
    return solution

def quantum_mpc(epsilon , de, C, Horizon, DT, algorithm = "CompressedVQE"):

    evs = len(epsilon)
    V=0.240
    Q  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
    Q1  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
    Q2  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
    #Q = get_qubomat3(11520,3,Horizon,V,DT)
    for ev in range(evs):
        Q[ev*Horizon*2:(ev+1)*Horizon*2,ev*Horizon*2:(ev+1)*Horizon*2] = get_qubomat3(epsilon[ev],de[ev],Horizon,V,DT)

    for evi in range(evs):
        for evj in range(evi,evs):
            Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = get_qubomat4(evi,evj,de[evi],de[evj],Horizon)

            #print(Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2])
    for evi in range(evs):
        for evj in range(evi,evs):
            Q2[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = get_qubomat_energy_limit1(evi,evj,de[evi],de[evj],Horizon, V, C)

    Q2[evs*Horizon*2: evs*Horizon*2 + Horizon*4, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = get_qubomat_energy_limit2(V, C, Horizon)
    #print(Q2[evs*Horizon*2: evs*Horizon*2 + Horizon*4, evs*Horizon*2 : evs*Horizon*2 + Horizon*4])
    #print(Q1)
    for ev in range(evs):
        Q2[ev*Horizon*2:(ev+1)*Horizon*2, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = get_qubomat_energy_limit3(de[ev], V,Horizon)
        #print(Q2[ev*Horizon*2:(ev+1)*Horizon*2, evs*Horizon*2 : evs*Horizon*2 + Horizon*4])
    Q = Q + 0.5*Q2
    if algorithm == "CompressedVQE":
        inst = CompressedVQE(Q,6,na=2)
        inst.optimize(n_measurements = 5000,number_of_experiments = 20,maxiter=500)
    else:
        inst = VQE(Q,2,na=2)
        inst.optimize(n_measurements = 5000,number_of_experiments = 20,maxiter=500)

    solution = inst.show_solution()

    solution = reshape_solution(solution,evs,Horizon)
    return ret_schedule(solution,evs,Horizon,de)