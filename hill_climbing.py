import random

def randomsolution(tsp):
    cities=list(range(len(tsp)))
    solution=[]

    for i in range(len(tsp)):
        randomcity=cities[random.randint(0,len(cities)-1)]
        solution.append(randomcity)
        cities.remove(randomcity)
    
    return solution

def routelen(tsp,solution):
    routelenght=0
    for i in range(len(solution)):
        routelenght+=tsp[solution[i-1]][solution[i]]
    return routelenght

def getneighbours(solution):
    neighbours=[]
    for i in range(len(solution)):
        for j in range(i+1,len(solution)):
            neighbour=solution.copy()
            neighbour[i]=solution[j]
            neighbour[j]=solution[i]
            neighbours.append(neighbour)
    return neighbours

def getbestneighbour(tsp,neighbours):
    bestroutelenght=routelen(tsp,neighbours[0])
    bestneighbour=neighbours[0]
    for neighbour in neighbours:
        currentroutelen=routelen(tsp,neighbour)
        if currentroutelen<bestroutelenght:
            bestroutelenght=currentroutelen
            bestneighbour=neighbour
    return bestneighbour,bestroutelenght

def hillclimbing(tsp):
    currentsolution=randomsolution(tsp)
    currentroutelen=routelen(tsp,currentsolution)
    neighbours=getneighbours(currentsolution)
    bestneighbour,bestneighbourlen=getbestneighbour(tsp,neighbours)

    while bestneighbourlen<currentroutelen:
        currentsolution=bestneighbour
        currentroutelen=bestneighbourlen
        neighbours=getneighbours(currentsolution)
        bestneighbour,bestneighbourlen=getbestneighbour(tsp,neighbours)

    return currentsolution,currentroutelen

tsp=[
    [0,400,500,300],
    [400, 0, 300, 500],
    [500, 300, 0, 400],
    [300, 500, 400, 0]
]

print(hillclimbing(tsp))