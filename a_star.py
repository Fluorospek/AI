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