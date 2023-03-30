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

    frontier = util.Stack()
    explored = set()
    frontier.push((current_node, (), 1))

    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_point, current_action, _ = current_node
        if problem.isGoalState(current_point):
            return list(current_action)
        if current_point not in explored:
            explored.add(current_point)
            for child in problem.getSuccessors(current_point):
                childPoint, childDirection, childCost = child
                if childPoint not in explored:
                    childDirection = current_action + (childDirection,)
                    frontier.push((childPoint, childDirection, childCost))


def extracDirectionsFromPath(path_order):
    res = []
    for action in path_order:
        if len(action) == 3:
            (_, nodeDirection, _) = action
            res.append(nodeDirection)
    return res

def getNodeXYPoint(node):
    if type(node) is tuple:
        if len(node) == 2:
            return node
        else:
            return node[0]
    else:
        return node

def buildReveredListFromPath(goalNode, path):
    currentNode = goalNode
    pathOrder = []
    while True:
        pathOrder.append(currentNode)
        currentPoint = getNodeXYPoint(currentNode)
        if currentPoint in path:
            currentNode = path[currentPoint]
        else:
            break
    pathOrder.reverse()
    return pathOrder

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    explored = set()
    frontier.push((problem.getStartState(), (), 1))

    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_point, current_action, _ = current_node
        if problem.isGoalState(current_point):
            return list(current_action)
        if current_point not in explored:
            explored.add(current_point)
            for child in problem.getSuccessors(current_point):
                childPoint, childDirection, childCost = child
                if childPoint not in explored:
                    childDirection = current_action + (childDirection,)
                    frontier.push((childPoint, childDirection, childCost))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    currentNude = problem.getStartState()
    currentNodeCost = 0

    frontier = util.PriorityQueue()
    explored = set()
    nodeToParentNode = {}
    nodeToNodeCost = {currentNude: currentNodeCost}
    while not frontier.isEmpty():
        currentNude = frontier.pop()
        currentPoint = getNodeXYPoint(currentNude)
        currentNodeCost = nodeToNodeCost[currentPoint]
        if problem.isGoalState(currentPoint):
            return buildSolution(currentNude, nodeToParentNode)
        explored.add(currentPoint)

        for child in problem.getSuccessors(currentPoint):
            childPoint, childDirection, childCost = child
            totalCost = childCost + currentNodeCost
            if childPoint not in explored and not isNodeExistsInPriorityQueue(frontier, childPoint):
                updatePathIfNeeded(childPoint, totalCost, currentNude, nodeToParentNode, nodeToNodeCost)
                frontier.push(child, childCost + currentNodeCost)
            elif existsPathWithHigherCost(frontier, childPoint, childCost):
                updatePathIfNeeded(childPoint, totalCost, currentNude, nodeToParentNode, nodeToNodeCost)
                frontier.update(child, childCost + currentNodeCost)

def buildSolution(current_nude, node_to_parent_node):
    (currentPoint, currentDirection, _) = current_nude
    pathOrder = buildReveredListFromPath(currentPoint, node_to_parent_node)
    return [*extracDirectionsFromPath(pathOrder), currentDirection]

def updatePathIfNeeded(child_point, total_cost, current_nude, node_to_parent_node,
                       node_cost_from_start_state):
    if child_point not in node_cost_from_start_state or \
            node_cost_from_start_state[child_point] > total_cost:
        node_cost_from_start_state[child_point] = total_cost
        node_to_parent_node[child_point] = current_nude

def isNodeExistsInPriorityQueue(priority_queue, childPoint):
    allItems = [entry[2] for entry in priority_queue.heap]
    # # item is a tuple of (point, direction, cost),
    # # we keep only those whose point is equal to the point we are looking for.
    allChildrenFilter = [x for x in allItems if x[0] == childPoint]
    return len(allChildrenFilter) > 0


def existsPathWithHigherCost(frontier, childPoint, childCost):
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
    currentNude = problem.getStartState()
    currentNodeCost = 0

    frontier = util.PriorityQueue()
    frontier.push(currentNude, currentNodeCost)
    explored = set()
    nodeToParentNode = {}
    nodeToNodeCost = {currentNude: currentNodeCost}
    while not frontier.isEmpty():
        currentNude = frontier.pop()
        currentPoint = getNodeXYPoint(currentNude)
        currentNodeCost = nodeToNodeCost[currentPoint]
        if problem.isGoalState(currentPoint):
            return buildSolution(currentNude, nodeToParentNode)
        explored.add(currentPoint)

        for child in problem.getSuccessors(currentPoint):
            childPoint, childDirection, childCost = child
            totalCost = childCost + currentNodeCost
            heuristicEstimation = heuristic(childPoint, problem)
            cost = totalCost + heuristicEstimation
            if childPoint not in explored and not isNodeExistsInPriorityQueue(frontier, childPoint):
                updatePathIfNeeded(childPoint, totalCost, currentNude, nodeToParentNode, nodeToNodeCost)
                frontier.push(child, cost)
            elif existsPathWithHigherCost(frontier, childPoint, childCost):
                updatePathIfNeeded(childPoint, totalCost, currentNude, nodeToParentNode, nodeToNodeCost)
                frontier.update(child, cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
