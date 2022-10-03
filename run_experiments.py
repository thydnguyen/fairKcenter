from time import time as time
import numpy as np
from scipy.spatial import distance
from kcenter import fairKcenter, fairKcenterPlusHeuristicB, HeuristicC
from baseline import Kleindessner, HeuristicA, HeuristicB
from copy import deepcopy

numIter = 10 #Number of iterations
percentEachGroup = 0.4/100 #Percentage of member for each group
PATH = r"" #Path to the dataset file in the dataset folder

def evaluate(X,C):
  return max(np.min(distance.cdist(X, X[C]),axis = 1))

def wrapper(iterN, func, *args):
    totalTime = []
    totalLoss = [] 
    for i in range(iterN):
        ARGS = deepcopy(args)
        start = time()
        C = func(*ARGS)
        elapse = time() - start
        loss = evaluate(ARGS[0], np.array(C).astype(int))
        if not np.array_equal(args[2] , np.unique(np.array(args[1])[C], return_counts=True)[1]):
            print("Fairness not satisfied")
        totalLoss.append(loss)
        totalTime.append(elapse)
    return totalLoss, totalTime

    


'''
Load the data from the npz file and set up the constraint vector to make sure the representation of each group
in the chosen centers to be approximately equal
'''
data = np.load(PATH)
X = data['x']
classTable  = data['y']
_, count = np.unique(data['y'], return_counts=True)
constraints = np.array([int(np.ceil(c* percentEachGroup)) for c in count])

objective, runtime = wrapper(numIter, fairKcenter ,X,classTable,constraints )
print("Our algorithm has mean objective value", np.mean(objective), "and", np.mean(runtime), "seconds runtime over", numIter, "runs")

objective, runtime = wrapper(numIter, Kleindessner ,X,classTable,constraints )
print("Kleindessner algorithm has mean objective value", np.mean(objective), "and", np.mean(runtime), "seconds runtime over", numIter, "runs")