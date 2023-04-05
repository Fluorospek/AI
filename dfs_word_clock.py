class State:
    def __init__(self, table):
        self.table = table

    def __str__(self):
        return str(self.table)

    def __eq__(self, other):
        return self.table == other.table

    def __hash__(self):
        return hash(str(self))


class BlocksWorld:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def is_goal(self, state):
        return state == self.goal_state

    def get_actions(self, state):
        actions = []
        for i in range(len(state.table)):
            for j in range(len(state.table)):
                if i != j and state.table[i] and not state.table[j]:
                    actions.append(('move', i, j))
        return actions

    def apply_action(self, state, action):
        action_type, i, j = action
        if action_type == 'move':
            new_table = [list(stack) for stack in state.table]
            new_table[j].append(new_table[i].pop())
            return State(new_table)

    def depth_first_search(self):
        visited = set()
        stack = [(State(self.initial_state), [])]
        while stack:
            state, path = stack.pop()
            if state in visited:
                continue
            visited.add(state)
            if self.is_goal(state.table):
                return path
            for action in self.get_actions(state):
                new_state = self.apply_action(state, action)
                if new_state:
                    stack.append((new_state, path + [action]))


initial_state = [[1], [3, 2]]
goal_state = [[], [3, 2, 1]]
bw = BlocksWorld(initial_state, goal_state)
solution = bw.depth_first_search()
print(solution)