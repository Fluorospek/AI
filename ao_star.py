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