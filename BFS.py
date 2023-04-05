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
