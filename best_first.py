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
