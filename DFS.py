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