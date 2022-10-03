import numpy as np
from scipy.spatial import distance
from hopcroftkarp import HopcroftKarp
from sklearn.metrics import pairwise_distances


def min_metric(x,X, metric = 'euclidean'):
  distance_matrix = distance.cdist(x, X, metric).flatten()
  return np.min(distance_matrix)


def HeuristicB(X,k, sexes,nr_centers_per_sex,given_centers, metric = 'euclidean'):
    '''Implementation of Heuristic B

    INPUT:
    sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers ... integer-vector with entries in 0,...,n-1

    RETURNS: heuristically chosen centers
    '''
    n = X.shape[0]
    d = X.shape[1]
    m = nr_centers_per_sex.size

    current_nr_per_sex=np.zeros(m)

    if k==0:
        cluster_centers = np.array([], dtype=int)
    else:
        if given_centers.size==0:
            cluster_centers=np.random.choice(n,1,replace=False)
            current_nr_per_sex[sexes[cluster_centers]]+=1
            kk=1
        else:
            cluster_centers = given_centers
            kk=0

        distance_to_closest = np.amin(pairwise_distances(X[cluster_centers].reshape(-1,d), X, metric = metric), axis=0)
        while kk<k:
            feasible_groups=np.where(current_nr_per_sex<nr_centers_per_sex)[0]
            feasible_points=np.where(np.isin(sexes,feasible_groups))[0]
            new_point=feasible_points[np.argmax(distance_to_closest[feasible_points])]
            current_nr_per_sex[sexes[new_point]] += 1
            cluster_centers = np.append(cluster_centers, new_point)
            distance_to_closest = np.amin(np.vstack((distance_to_closest, pairwise_distances(X[new_point].reshape(-1,d), X, metric))), axis=0)
            kk+=1

        cluster_centers=cluster_centers[given_centers.size:]

    return cluster_centers


def gonzalez(X,k, metric = 'euclidean'): 
  '''  
  X :the data set, 2d-aray
  k : is the number of cluster, unsigned int
  RETURNS: list of potential unfair centers
  '''
  C = [] #list of centers to return
  C.append(np.random.randint(0, X.shape[0]))
  K = 1
  kDistance = [] #table storing distance of k centers to other points
  minDist = distance.cdist(X[C], X, metric).flatten()
  kDistance.append(minDist)
  while  k!=K :
    candidate = np.argmax(minDist)
    C.append(candidate)
    K = K+1
    newDist = distance.cdist([X[candidate]], X, metric ).flatten()
    kDistance.append(newDist)
    if k!= K:
      minDist = np.min(np.vstack((minDist, newDist)), axis = 0)
  return C, kDistance


def gonzalez_variant(X,candidates, k, given_dmat, metric= 'euclidean'): 
  '''  
  X :the data set, 2d-aray
  k : is the number of cluster, unsigned int
  given_dmat: distance matrice that's already computed
  candidates: list of indices considered for greedy selection
  RETURNS: list of potential unfair centers
  '''
  X_sub = X[candidates]
  
  C = [] #list of centers to return
  given_dmat_min = np.min(given_dmat, axis = 0)
  candidate_given_dmat_min = given_dmat_min[candidates]
  C.append(candidates[np.argmax(candidate_given_dmat_min)])
  
  minDist = distance.cdist(X[C], X, metric).flatten()
  candidate_minDist = minDist[candidates]
  if  k == 1: 
    return C,  np.concatenate((given_dmat , [minDist]), axis = 0) 

  K = 1
  kDistance = []  
  candidate_minDist = np.min(np.vstack((candidate_minDist, candidate_given_dmat_min)), axis = 0)
  kDistance.append(minDist)
  while  k!=K :
    candidate = np.argmax(candidate_minDist)
    C.append(candidates[candidate])
    K = K+1
    newDist = distance.cdist([X_sub[candidate]], X, metric ).flatten()
    kDistance.append(newDist)
    if k!= K:
      candidate_minDist = np.min(np.vstack((candidate_minDist, newDist[candidates])), axis = 0)
  return C,  np.concatenate((given_dmat , kDistance), axis = 0) 


def testFairShift(X, neighbors, distance, constraints, minDist):
  '''
  X: list of indices that we want to test fair-shift constraint
  neighbor: a table of size len(X) X m, that stores the nearest neighbor of each group for each point in X
  distance:  a table of size len(X) X m, that stores distance to the nearest neighbor of each group for each point in X
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  minDist: minimum distance to be considered for fairshift
  RETURNS: False if the fairshift constraint cannot be satisfied otherwise the list of indices that satisfies the constraint
  '''
  possibleEdges = dict()
  lenX = len(X)
  for d,n,i in zip(distance, neighbors, range(len(distance))):
      possibleEdges[i] = set( [j  for j in range(lenX) if d[j] < minDist] )

  graph = dict()
  
  for i, j in possibleEdges.items():
    jEdges = []
    for jj in j:
      temp = [str(jj) + "|" + str(c) for c in  range(constraints[jj])]
      jEdges.extend(temp)
    graph[i] = tuple(jEdges)
  maxMatch = HopcroftKarp(graph).maximum_matching(keys_only=True)
  if len(maxMatch) != len(distance):
    return False
  else:
    result =  [neighbors[i][int(maxMatch[i].split("|")[0])] for i in range(len(X))]
    return result


def findAllNeighbors(classTable, M, kDistance_i):
  '''
  lassTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k
  M: number of groups
  kDistance_i: vector containing the distance of the center i to all other points
  RETURNS: a list object that closest to center i to each group and a list that contains the corresponding distance
  '''
  
  distTable = [float('infinity')] * M
  neighborTable = [-1] * M
  for D, currentGroup,m in zip(kDistance_i, classTable, list(range(len(classTable)))):
      if D < distTable[currentGroup]:
          distTable[currentGroup] = D
          neighborTable[currentGroup] = m

  return neighborTable, distTable

def HeuristicC(X, classTable, constraints, metric = 'euclidean'):
  """
  Implementation of Heuristic C
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k 
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
  k = np.sum(constraints)
  kDistance = []
  M = len(constraints)
  candidate = np.random.randint(len(X))
  fairshift = [candidate]
  k = k - 1
  kDistance = distance.cdist([X[candidate]], X, metric )
  constraints[classTable[candidate]] = constraints[classTable[candidate]]  - 1
  for i in range(M):
      if constraints[i] == 0:
          continue
      else:
          classTableK = [x for x in range(len(classTable)) if classTable[x] == i] 
          candidate = np.array(list(set(classTableK) - set(fairshift)))
          addK, kDistance = gonzalez_variant(X,candidate,constraints[i], kDistance, metric=metric )
          fairshift.extend(addK)
  return fairshift


def fairKcenter(X, classTable, constraints, metric = 'euclidean'):
  """
  Implementation of Alg2-Seq
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k 
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
  k = np.sum(constraints)
  classTable = classTable.tolist()
  unfairCenters, kDistance = gonzalez(X,k, metric = metric)
  kDistance = np.array(kDistance).tolist()
  M = len(classTable)
  neighborTable = [None] * k
  distTable = [None] * k
  first = 0
  last = k - 1
  fairshift = None
  bestMid = -float('Infinity')
  bestRadius = float('Infinity')
  bestFairShift = None
  old_mid = 0

  while( first<=last):
    mid = (first + last)//2
    for i in range(old_mid, mid+1):
      if i < k and distTable[i] == None:
          neighborTable[i], distTable[i] = findAllNeighbors(classTable, M, kDistance[i])
    if mid > 0:
      minDist = min_metric([X[unfairCenters[mid]]] , X[unfairCenters[:mid]]) / 2
      fairshift = testFairShift(unfairCenters[:mid + 1 ], neighborTable[:mid + 1], distTable[:mid  + 1], constraints, minDist)
    else:
      fairshift = True
    if fairshift == False:
      last = mid - 1
    else:
      first = mid + 1
      bestMid = mid
      bestRadius = 	minDist
      bestFairShift = np.copy(fairshift)
    old_mid = max([mid,old_mid])
  
  mid = bestMid

  minDist = bestRadius
  candidateRadius = sorted([r for r in np.ravel([distTable[:mid + 1]]) if r <= minDist])
  bestMinDist = minDist
  
  first = 0
  last = len(candidateRadius) - 1  
  fairshift = None

  while (first <= last):
    midRadius = (first + last) // 2
    minDist = candidateRadius[midRadius]
    fairshift = testFairShift(unfairCenters[:mid+1], neighborTable[:mid+1], distTable[:mid+1], constraints, minDist)
    if fairshift != False and minDist <= bestMinDist:
      bestMinDist = minDist
      bestFairShift = fairshift[:]
      last = midRadius - 1	
    else:
      first = midRadius + 1	

  classTable = np.array(classTable)
  fairshift = bestFairShift[:]
  constraintsSatisfied, constraintsSatisfiedCount = np.unique(np.array(classTable)[fairshift], return_counts = True)
  for c in range(len(constraintsSatisfied)):
    constraints[constraintsSatisfied[c]] = constraints[constraintsSatisfied[c]] - constraintsSatisfiedCount[c]
  
  if len(fairshift) == k:
      return fairshift
  for i in range(len(classTable)):
    if i not in fairshift and constraints[classTable[i]] > 0:      
      fairshift.append(i)
      constraints[classTable[i]]  =  constraints[classTable[i]]  - 1
      if len(fairshift) == k:
         break
    
  return fairshift

def fairKcenterPlusHeuristicB(X, classTable, constraints, metric = 'euclidean'):
  """
  Implementation of Alg2-HeuristicB
  X: dataset
  classTable: class assignment for each point, integer-vector of length m with entries in 0,...,k and sum over entries equaling k 
  constraints: required count for each group  , integer-vector with entries in 0,...,n-1
  RETURNS: list of fair centers
  """
  k = np.sum(constraints)
  classTable = classTable.tolist()
  unfairCenters, kDistance = gonzalez(X,k, metric = metric)
  kDistance = np.array(kDistance).tolist()
  M = len(classTable)
  neighborTable = [None] * k
  distTable = [None] * k
  first = 0
  last = k - 1
  fairshift = None
  bestMid = -float('Infinity')
  bestRadius = float('Infinity')
  bestFairShift = None
  old_mid = 0

  while( first<=last):
    mid = (first + last)//2
    for i in range(old_mid, mid+1):
      if i < k and distTable[i] == None:
          neighborTable[i], distTable[i] = findAllNeighbors(classTable, M, kDistance[i])
    if mid > 0:
      minDist = min_metric([X[unfairCenters[mid]]] , X[unfairCenters[:mid]]) / 2
      fairshift = testFairShift(unfairCenters[:mid + 1 ], neighborTable[:mid + 1], distTable[:mid  + 1], constraints, minDist)
    else:
      fairshift = True
    if fairshift == False:
      last = mid - 1
    else:
      first = mid + 1
      bestMid = mid
      bestRadius = 	minDist
      bestFairShift = np.copy(fairshift)
    old_mid = max([mid,old_mid])
  
  mid = bestMid

  minDist = bestRadius
  candidateRadius = sorted([r for r in np.ravel([distTable[:mid + 1]]) if r <= minDist])
  bestMinDist = minDist
  
  first = 0
  last = len(candidateRadius) - 1  
  fairshift = None

  while (first <= last):
    midRadius = (first + last) // 2
    minDist = candidateRadius[midRadius]
    fairshift = testFairShift(unfairCenters[:mid+1], neighborTable[:mid+1], distTable[:mid+1], constraints, minDist)
    if fairshift != False and minDist <= bestMinDist:
      bestMinDist = minDist
      bestFairShift = fairshift[:]
      last = midRadius - 1	
    else:
      first = midRadius + 1	

  classTable = np.array(classTable)
  fairshift = bestFairShift[:]
  constraintsSatisfied, constraintsSatisfiedCount = np.unique(np.array(classTable)[fairshift], return_counts = True)
  for c in range(len(constraintsSatisfied)):
    constraints[constraintsSatisfied[c]] = constraints[constraintsSatisfied[c]] - constraintsSatisfiedCount[c]
  
  if len(fairshift) == k:
      return fairshift
  for i in range(len(classTable)):
    if i not in fairshift and constraints[classTable[i]] > 0:      
      fairshift.append(i)
      constraints[classTable[i]]  =  constraints[classTable[i]]  - 1
      if len(fairshift) == k:
         break
  
  fairshift_ = HeuristicB(X,k-len(fairshift), classTable, np.array(constraints), np.array(fairshift))
  fairshift = np.concatenate((fairshift, fairshift_), axis = None)
  return fairshift
