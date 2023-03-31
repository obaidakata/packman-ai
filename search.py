# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import traceback

import util
from game import Directions


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    current_node = problem.getStartState()
    if problem.isGoalState(current_node):
        return []

    # LIFO data structure, contains the nodes which will be explored.
    frontier = util.Stack()
    frontier.push(current_node)
    # contains the nodes that already explored.
    explored = set()
    # for node v, node_to_parent_node[v] returns the parent of v that helped us to explore v.
    node_to_parent_node = {}
    while not frontier.isEmpty():
        current_node = frontier.pop()
        # extract the point (x,y) from the node
        current_point = get_node_point(current_node)
        explored.add(current_point)

        # check whether the current point is a goal
        if problem.isGoalState(current_point):
            # return a list of actions required to traverse from the root to the goal.
            return build_solution(current_node, node_to_parent_node)

        for child in problem.getSuccessors(current_point):
            (child_point, _, _) = child
            if child_point not in explored:
                # set current_node as the parent of the child node
                node_to_parent_node[child_point] = current_node
                frontier.push(child)

def extracDirectionsFromPath(path_order):
    res = []
    for action in path_order:
        if len(action) == 3:
            (_, nodeDirection, _) = action
            res.append(nodeDirection)
    return res

def get_node_point(node):
    if type(node) is tuple:
        if len(node) == 2:
            # node is (x, y)
            return node
        else:
            # node is ((x, y), direction, cost)
            return node[0]
    else:
        return node

def buildReveredListFromPath(goal_node, path):
    current_node = goal_node
    path_order = []
    while True:
        path_order.append(current_node)
        current_point = get_node_point(current_node)
        if current_point in path:
            current_node = path[current_point]
        else:
            break
    path_order.reverse()
    return path_order

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    current_node = problem.getStartState()
    if problem.isGoalState(current_node):
        return []

    # FIFO data structure, contains the nodes which will be explored.
    frontier = util.Queue()
    frontier.push(current_node)
    # contains the nodes that already explored.
    explored = set()
    # for node v, node_to_parent_node[v] returns the parent of v that helped us to explore v.
    node_to_parent_node = {}
    while not frontier.isEmpty():
        current_node = frontier.pop()
        # extract the point (x,y) from the node
        current_point = get_node_point(current_node)
        explored.add(current_point)

        # check whether the current point is a goal
        if problem.isGoalState(current_point):
            # return a list of actions required to traverse from the root to the goal.
            return build_solution(current_node, node_to_parent_node)

        for child in problem.getSuccessors(current_point):
            (child_point, _, _) = child
            # we check whether the child_point is already in node_to_parent_node because BFS the first path we found
            # is shortest.
            if child_point not in explored and child_point not in node_to_parent_node:
                # set current_node as the parent of the child node
                node_to_parent_node[child_point] = current_node
                frontier.push(child)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    current_node = problem.getStartState()
    current_node_cost = 0
    # contains the nodes which will be explored, sorted by least cost.
    frontier = util.PriorityQueue()
    frontier.push(current_node, current_node_cost)
    # contains the nodes that already explored.
    explored = set()
    # for node v, node_to_parent_node[v] returns the parent of v that helped us to explore v.
    node_to_parent_node = {}
    # for node v, node_to_node_cost[v] returns the cost of v.
    node_to_node_cost = {current_node: current_node_cost}
    while not frontier.isEmpty():
        current_node = frontier.pop()
        # extract the point (x,y) from the node
        current_point = get_node_point(current_node)
        # extract the cost of current_node
        current_node_cost = node_to_node_cost[current_point]
        explored.add(current_point)

        # check whether the current point is a goal
        if problem.isGoalState(current_point):
            # return a list of actions required to traverse from the root to the goal.
            return build_solution(current_node, node_to_parent_node)

        for child in problem.getSuccessors(current_point):
            child_point, _, child_cost = child
            # the cost of node v equal to the cost of the path from starting node to the parent of v
            # plus the cost of the edge (parent_v, v).
            total_cost = child_cost + current_node_cost
            if child_point not in explored and not do_node_exists_in_priorityQueue(frontier, child_point):
                # update the path if we find a cheaper path from the starting point to the child.
                update_path_if_needed(child_point, total_cost, current_node, node_to_parent_node, node_to_node_cost)
                frontier.push(child, total_cost)
            elif do_node_exists_with_higher_cost(frontier, child_point, child_cost):
                # update the path if we find a cheaper path from the starting point to the child.
                update_path_if_needed(child_point, total_cost, current_node, node_to_parent_node, node_to_node_cost)
                frontier.update(child, total_cost)

# builds the actions required to traverse from the root to the goal node.
def build_solution(current_nude, node_to_parent_node):
    (currentPoint, currentDirection, _) = current_nude
    pathOrder = buildReveredListFromPath(currentPoint, node_to_parent_node)
    return [*extracDirectionsFromPath(pathOrder), currentDirection]

#
def update_path_if_needed(child_point, total_cost, current_node, node_to_parent_node,
                          node_cost_from_start_state):
    # checks if we calculated the cost of the path from stating point before
    should_update_path = child_point not in node_cost_from_start_state
    # checks if the cost of the path from the starting_node to the child_node that passes through the current_node
    # is cheaper than the previews path we found from starting_node to the child_node
    should_update_path = should_update_path or node_cost_from_start_state[child_point] > total_cost
    if should_update_path:
        # if yes update the path and the path cost.
        node_cost_from_start_state[child_point] = total_cost
        node_to_parent_node[child_point] = current_node
    return should_update_path
def do_node_exists_in_priorityQueue(priority_queue, childPoint):
    allItems = [entry[2] for entry in priority_queue.heap]
    # # item is a tuple of (point, direction, cost),
    # # we keep only those whose point is equal to the point we are looking for.
    allChildrenFilter = [x for x in allItems if x[0] == childPoint]
    return len(allChildrenFilter) > 0

def do_node_exists_with_higher_cost(frontier, childPoint, childCost):
    # # frontier.heap store tuples entries (priority, count, item), we extract the item.
    allItems = [entry[2] for entry in frontier.heap]
    # # item is a tuple of (point, direction, cost),
    # # we keep only items whose point is equal to the point we are looking for.
    allChildrenFilter = [x for x in allItems if x[0] == childPoint]
    if len(allChildrenFilter) > 0:
        oldChildCost = allChildrenFilter[0][2]
        return oldChildCost > childCost
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    current_node = problem.getStartState()
    current_node_cost = 0
    # contains the nodes which will be explored, sorted by least cost.
    frontier = util.PriorityQueue()
    frontier.push(current_node, current_node_cost)
    # contains the nodes that already explored.
    explored = set()
    # for node v, node_to_parent_node[v] returns the parent of v that helped us to explore v.
    node_to_parent_node = {}
    # for node v, node_to_node_cost[v] returns the cost of v.
    node_to_node_cost = {current_node: current_node_cost}
    while not frontier.isEmpty():
        current_node = frontier.pop()
        # extract the point (x,y) from the node
        current_point = get_node_point(current_node)
        # extract the cost of current_node
        current_node_cost = node_to_node_cost[current_point]
        explored.add(current_point)

        # check whether the current point is a goal
        if problem.isGoalState(current_point):
            # return a list of actions required to traverse from the root to the goal.
            return build_solution(current_node, node_to_parent_node)

        for child in problem.getSuccessors(current_point):
            child_point, _, child_cost = child
            total_cost = child_cost + current_node_cost
            heuristic_estimation = heuristic(child_point, problem)
            # the cost of node (v) is equal to the cost of the path from starting node to the parent of v
            # plus the cost of the edge (v_parent, v)
            # plus the heuristic estimation of v to the goal.
            # cost(v) = cost(starting_point ~> v_parent) + cost(edge(v_parent, v)) + heuristic(v)
            cost = total_cost + heuristic_estimation
            if child_point not in explored and not do_node_exists_in_priorityQueue(frontier, child_point):
                # update the path if we find a cheaper path from the starting point to the child.
                update_path_if_needed(child_point, total_cost, current_node, node_to_parent_node, node_to_node_cost)
                frontier.push(child, cost)
            elif do_node_exists_with_higher_cost(frontier, child_point, child_cost):
                # update the path if we find a cheaper path from the starting point to the child.
                update_path_if_needed(child_point, total_cost, current_node, node_to_parent_node, node_to_node_cost)
                frontier.update(child, cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
