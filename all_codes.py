#bfs
from collections import defaultdict


class graph:
    def __init__(self):
        self.new_dict = defaultdict(list)

    def insert(self, key, value):
        self.new_dict[key].append(value)

    def print_ele(self):
        print(self.new_dict)

    def bfs_trav(self, init):
        visited = [False]*(len(self.new_dict))
        # false list for all nodes,as none of the have been visited
        queue = []
        queue.append(init)
        visited[init] = True

        while queue:
            node = queue.pop(0)
            print('{}->'.format(node), end=" ")

            # inserting all children nodes of the node recently popped
            for ele in self.new_dict[node]:
                if visited[ele] == False:
                    queue.append(ele)
                    visited[ele] = True


graph_bfs = graph()
graph_bfs.insert(2, 0)
graph_bfs.insert(2, 3)
graph_bfs.insert(0, 1)
graph_bfs.insert(0, 2)
graph_bfs.insert(3, 3)
graph_bfs.insert(1, 2)
graph_bfs.print_ele()

graph_bfs.bfs_trav(0)


#dfs
from collections import defaultdict


class graph:
    def __init__(self):
        self.new_dict = defaultdict(list)

    def insert(self, key, value):
        self.new_dict[key].append(value)

    def print_ele(self):
        print(self.new_dict)

    def dfs_trav(self, visited, node):
        visited.add(node)
        print('{}->'.format(node), end=" ")

        for ele in self.new_dict[node]:
            if ele not in visited:
                self.dfs_trav(visited, ele)

    def dfs(self, init):
        visited = set()

        self.dfs_trav(visited, init)


graph_bfs = graph()
graph_bfs.insert(2, 0)
graph_bfs.insert(2, 3)
graph_bfs.insert(0, 1)
graph_bfs.insert(0, 2)
graph_bfs.insert(3, 3)
graph_bfs.insert(1, 2)
graph_bfs.print_ele()

x=int(input('Enter starting node'))
graph_bfs.dfs(x)

#best_first
from queue import PriorityQueue
n=int(input("Enter number of vertices:\n"))
graph=[[] for i in range(n)]

def best_first(src, goal, n):
    visited=[False]*n
    pq=PriorityQueue()
    pq.put((0,src))
    visited[src]=True

    while pq.empty()==False:
        u=pq.get()[1]
        print(u,end=" ")
        if u==goal:
            break

        for v,wt in graph[u]:
            if visited[v]==False:
                visited[v]=True
                pq.put((wt,v))

def add_edge(x,y,wt):
    graph[x].append((y,wt))
    graph[y].append((x,wt))

for i in range(n):
    x=int(input())
    y=int(input())
    wt=int(input())
    add_edge(x,y,wt)

src=int(input("Enter source vertex:\n"))
goal=int(input("Enter goal vertex:\n"))
best_first(src,goal,n)

#hill climbing
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

#a star
from queue import PriorityQueue

def aStart(src,goal):
    open=set(src)
    closed=set()
    g={}
    parents={}
    g[src]=0
    parents[src]=src

    while len(open)>0:
        n=None

        for v in open:
            if n==None or g[v]+heuristic(v)<g[n]+heuristic(n):
                n=v
            
        if n==goal or graph[n]==None:
            pass
        else:
            for (m,wt) in get_neighbour(n):
                if m not in open and m not in closed:
                    open.add(m)
                    parents[m]=n
                    g[m]=g[n]+wt
                else:
                    if g[m]>g[n]+wt:
                        g[m]=g[n]+wt
                        parents[m]=n

                        if m in closed:
                            closed.remove(m)
                            open.add(m)
            
        if n==None:
            print('Path does not exist')
            return None
            
        if n==goal:
            path=[]

            while parents[n]!=n:
                path.append(n)
                n=parents[n]
                
            path.append(src)

            path.reverse()

            print('Path found: {}'.format(path))
            return path

        open.remove(n)
        closed.add(n)

    print('Path does not exist')
    return None
    
def get_neighbour(v):
    if v in graph:
        return graph[v]
    else:
        return None
    
def heuristic(n):
    H={
        'A':11,
        'B':6,
        'C':99,
        'D':1,
        'E':7,
        'G':0
    }

    return H[n]
    
graph={
    'A':[('B',2),('E',3)],
    'B':[('C',1),('G',9)],
    'C':None,
    'E':[('D',6)],
    'D':[('G',1)],
}

aStart('A','G')

#ao star
def Cost(H, condition, weight=1):
    cost={}
    if 'AND' in condition:
        AND_nodes=condition['AND']
        path_A=' AND '.join(AND_nodes)
        pathA=sum(H[node]+weight for node in AND_nodes)
        cost[path_A]=pathA
    if 'OR' in condition:
        OR_nodes=condition['OR']
        path_B=' OR '.join(OR_nodes)
        pathB=min(H[node]+weight for node in OR_nodes)
        cost[path_B]=pathB
    return cost

def update_cost(H,Conditions, weight=1):
    main_nodes=list(Conditions.keys())
    main_nodes.reverse()
    least_cost={}
    for key in main_nodes:
        condition=Conditions[key]
        print(key,':',Conditions[key],'>>',Cost(H,condition,weight))
        c=Cost(H,condition,weight)
        least_cost[key]=Cost(H,condition,weight)
    return least_cost

def shortest_path(src,Updated_cost,H):
    path=src
    if src in Updated_cost.keys():
        min_cost=min(Updated_cost[src].values())
        key=list(Updated_cost[src].keys())
        values=list(Updated_cost[src].values())
        index=values.index(min_cost)
        next=key[index].split()
        if len(next)==1:
            src=next[0]
            path+=' = '+shortest_path(src,Updated_cost,H)
        else:
            path+='=('+key[index]+') '
            src=next[0]
            path+='['+shortest_path(src,Updated_cost,H)+' + '
            src=next[-1]
            path+=shortest_path(src,Updated_cost,H)+']'
    return path


H1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
Conditions = {
 'A': {'OR': ['D'], 'AND': ['B', 'C']},
 'B': {'OR': ['G', 'H']},
 'C': {'OR': ['J']},
 'D': {'AND': ['E', 'F']},
 'G': {'OR': ['I']}
}
weight = 1
print('Updated Cost :')
Updated_cost = update_cost(H1, Conditions, weight=1)
print('Shortest Path: \n',shortest_path('A',Updated_cost,H1))

#8 puzzle
import copy

class Node:
    def __init__(self,data,level,fval):
        self.data=data
        self.level=level
        self.fval=fval

    def generate_child(self):
        # genrating child nodes from the given state by moving the blank space either of the four directions
        x,y=self.find(self.data,'_')
        #val_list contains position values for moving the blank space in either of the 4 directions [up,down,left,right] respectively
        val_list=[[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children=[]
        for i in val_list:
            child=self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node=Node(child,self.level+1,0)
                children.append(child_node)
        return children
    
    def shuffle(self,puz,x1,y1,x2,y2):
        #move blank space in given directions and check if it went out of bounds, return none if out of bound
        if x2>=0 and x2< len(self.data) and y2>=0 and y2< len(self.data):
            temp_puz=[]
            temp_puz=self.copy(puz)
            temp=temp_puz[x2][y2]
            temp_puz[x2][y2]=temp_puz[x1][y1]
            temp_puz[x1][y1]=temp
            return temp_puz
        else:
            return None
        
    def copy(self,root):
        #copy function to create a similar matrix of the given node
        temp=[]
        for i in root:
            t=[]
            for j in i:
                t.append(j)
            temp.append(t)
        return temp

    def find(self,puz,x):
        #to find the position of blank space
        for i in range(0, len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j

class Puzzle:
    def __init__(self, size):
        self.n=size
        self.open=[]
        self.closed=[]

    def accept(self):
        puz=[]
        for i in range (0,self.n):
            temp=input().split(" ")
            puz.append(temp)
        return puz
    
    def h(self, start, goal):
        # calculate the difference between the given puzzles=
        temp=0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp+=1
        return temp


    def f(self, start,goal):
        #heuristic function to calculate heuristic value f(x)=g(x)+h(x)
        return self.h(start.data,goal)+start.level

    def process(self):
        print('Enter the start state')
        start=self.accept()
        print('Enter the goal state')
        goal=self.accept()
        start=Node(start,0,0)
        start.fval=self.f(start,goal)
        # append start into open node
        self.open.append(start)
        print('\n\n')
        while True:
            cur=self.open[0]
            print("####################################################")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            # if diff between start node and goal node is 0, we terminate the loop
            if(self.h(cur.data,goal)==0):
                break
            for i in cur.generate_child():
                i.fval=self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]
            #sorting the open list based on f values
            self.open.sort(key=lambda x:x.fval, reverse=False)


puz=Puzzle(3)
puz.process()

#travelling salesman problem
from sys import maxsize
v=4
def travel(graph,s):
    vertex=[]
    for i in range(v):
        if i!=s:
            vertex.append(i)
    minpath=maxsize
    while True:
        present_cost=0
        k=s
        for i in range(len(vertex)):
            present_cost+=graph[k][vertex[i]]
            k=vertex[i]  
        present_cost+=graph[k][s]
        minpath=min(minpath,present_cost)
        if not permutations(vertex):
            print("The path followed is:",vertex[::-1])
            break
    return minpath
def permutations(l):
        a=len(l)
        i=a-2
        while (i>=0 and l[i]>l[i+1]):
            i-=1   
        if i==-1:
            return False        
        j=i+1
        while j<a and l[j]>l[i]:
            j+=1
        j-=1
        l[i],l[j]=l[j],l[i]
        left=i+1
        right=a-1
        while left<right:
            l[left],l[right]=l[right],l[left]
            left+= 1
            right-=1
        return True

graph=[[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
s=int(input("Enter the starting node:"))
travel(graph,s)

#water jug
from queue import Queue
def nextState(current,max_x,max_y):
    states = []
    states.append((max_x,current[1]))
    states.append((current[0],max_y))
    states.append((0,current[1]))
    states.append((current[0],max_y))
    pour_x_y = min(current[0],max_y-current[1])
    states.append((current[0]-pour_x_y,current[1]+pour_x_y))
    pour_y_x = min(current[1],max_x-current[0])
    states.append((current[0]+pour_y_x,current[1]-pour_y_x))
    if(states.index(current)):
        states.remove(current)
    return set(states)
def bfs(start,goal,max_x,max_y):
    q = Queue()
    q.put(start)
    visited = set()
    while q:
        current = q.get()
        if current==goal:
            return True
        visited.add(current)
        States = nextState(current,max_x,max_y)
        print("Current State ",current)
        print("Next States")
        for next in States:
            if next not in visited:
                q.put(next)
                print(next)
    return False
start_state = (0,0)
goal_state = (2,0)
max_x = 4
max_y = 3
is_reaxhable = bfs(start_state,goal_state,max_x,max_y)
print(is_reaxhable)

