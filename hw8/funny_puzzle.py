import heapq
import numpy as np
import copy


def get_matrices(state):
    state = np.array(state)
    state = state.reshape(3, 3)
    return state

def isValidPos(x, y):
    if (x < 0 or y < 0 or x > 2 or y > 2):
        return 0
    return 1

def get_adjacent_tiles(x, y, state):
    v = []
    if (isValidPos(x - 1, y)):
        if state[x - 1][y] != 0:
            v.append(((x - 1), y))
    if (isValidPos(x, y - 1)):
        if state[x][y - 1] != 0:
            v.append((x, (y - 1)))
    if (isValidPos(x, y + 1)):
        if state[x][y + 1] != 0:
            v.append((x, (y + 1)))
    if (isValidPos(x + 1, y)):
        if state[x + 1][y] != 0:
            v.append(((x + 1), y))
    return v



def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    from_state = get_matrices(from_state)
    to_state = get_matrices(to_state)
    distance = 0
    for i in range(0, 3):
        for j in range(0, 3):
            item = from_state[i][j]
            if item == 0 or to_state[i][j] == from_state[i][j]:
                continue
            val = np.where(to_state == item)
            x = val[0][0]
            y = val[1][0]
            distance = distance + (abs(x - i) + abs(y - j))
    return distance




def print_succ(state):
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    state = get_matrices(state)
    val = np.where(state == 0)
    x1 = val[0][0]
    y1 = val[1][0]
    x2 = val[0][1]
    y2 = val[1][1]
    succ_states = []
    state1 = get_adjacent_tiles(x1, y1, state)
    state2 = get_adjacent_tiles(x2, y2, state)

    for i in state1:
        inter_state = copy.deepcopy(state)
        inter_state[x1][y1] = state[i[0]][i[1]]
        inter_state[i[0]][i[1]] = 0
        inter_state = inter_state.reshape(1, 9)
        succ_states.append(list(inter_state[0]))
    for i in state2:
        inter_state = copy.deepcopy(state)
        inter_state[x2][y2] = state[i[0]][i[1]]
        inter_state[i[0]][i[1]] = 0
        inter_state = inter_state.reshape(1, 9)
        succ_states.append(list(inter_state[0]))
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    open = []
    closed = []
    parent_index = -1
    final_path = dict()
    g = 0
    h = get_manhattan_distance(state, goal_state)
    cost = g + h
    uid = 0
    heapq.heappush(open, (cost, state, (g, h, uid, parent_index)))
    while len(open) > 0:
        n = heapq.heappop(open)
        closed.append(n[1])
        final_path[n[2][2]] = []
        final_path[n[2][2]].append((n[1], n[2][1], n[2][0], n[2][3]))
        if n[1] == goal_state:
            break
        parent_index = n[2][2]
        succ = get_succ(n[1])
        for i in succ:
            if i in closed:
                continue
            else:
                g = n[2][0] + 1
                h = get_manhattan_distance(i, goal_state)
                cost = g + h
                uid += 1
                heapq.heappush(open, (cost, i, (g, h, uid, parent_index)))

    max_queue_length = len(open)
    path = []
    last_key = list(final_path.keys())[-1]
    while last_key != -1:
        path.append(final_path[last_key])
        parent = final_path[last_key][0][3]
        last_key = parent
    path_reversed = list(reversed(path))
    for i in path_reversed:
        print(f'{i[0][0]} h={i[0][1]} moves: {i[0][2]}')
    print("Max queue length:", max_queue_length + 1)

if __name__ == "__main__":
    pass
