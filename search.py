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

    currentNude = problem.getStartState()
    if problem.isGoalState(currentNude):
        return []

    frontier = util.Stack()
    frontier.push(currentNude)
    explored = set()
    path = {}
    while not frontier.isEmpty():
        currentNude = frontier.pop()
        currentPoint = getNodeXYPoint(currentNude)
        explored.add(currentPoint)

        if problem.isGoalState(currentPoint):
            currentDirection = currentNude[1]
            pathOrder = buildReveredListFromPath(currentPoint, path)
            solution = [*extracDirectionsFromPath(pathOrder), currentDirection]
            return solution

        for child in problem.getSuccessors(currentPoint):
            childPoint, childDirection, childCost = child
            # TODO: check why checking is childPoint is already in path gives wrong solutions (1)
            if childPoint not in explored and child not in frontier.list:
                path[childPoint] = currentNude
                frontier.push(child)


# def draft(problem): # doesn't work
#
#     currentNude = problem.getStartState()
#     if problem.isGoalState(currentNude):
#         return []
#
#     frontier = util.Stack()
#     frontier.push(currentNude)
#     explored = set()
#     path = {}
#     while not frontier.isEmpty():
#         currentNude = frontier.pop()
#         explored.add(getNodeXYPoint(currentNude))
#         for child in problem.getSuccessors(getNodeXYPoint(currentNude)):
#             childPoint, childDirection, childCost = child
#             if childPoint not in explored and child not in frontier.list:
#                 path[childPoint] = currentNude
#                 if problem.isGoalState(childPoint):
#                     pathOrder = buildReveredListFromPath(childPoint, path)
#                     solution = [*extracDirectionsFromPath(pathOrder), childDirection]
#                     # print(solution)
#                     return solution
#                 frontier.push(child)

def extracDirectionsFromPath(path_order):
    res = []
    for action in path_order:
        if len(action) == 3:
            res.append(getNodeDirection(action))
    return res

def getNodeDirection(node):
    return node[1]
def getNodeXYPoint(node):
    if type(node) is tuple:
        if len(node) == 2:
            return node
        else:
            return node[0]
    else:
        return node


def printGraph(problem):
    if problem is not None:
        explored = set()
        printNode(problem.getStartState(), problem, explored)

def printNode(node, problem, explored):
    if node is not None:
        print('node', node)
        explored.add(node)
        nodeSuccessors = problem.getSuccessors(node)
        print(nodeSuccessors)
        for node in nodeSuccessors:
            # print('node[0]', node[0], 'successors', problem.getSuccessors(node[0]))
            if node[0] not in explored:
                printNode(node[0], problem, explored)
            # else:
            #     print(node[0], 'is in explored', explored)

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


def listToQueue(items):
    queue = util.Queue()
    for item in items:
        queue.push(item)
    return queue


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    currentNude = problem.getStartState()
    if problem.isGoalState(currentNude):
        return []

    frontier = util.Queue()
    frontier.push(currentNude)
    explored = set()
    path = {}
    while not frontier.isEmpty():
        currentNude = frontier.pop()
        currentPoint = getNodeXYPoint(currentNude)
        explored.add(currentPoint)

        if problem.isGoalState(currentPoint):
            currentDirection = currentNude[1]
            pathOrder = buildReveredListFromPath(currentPoint, path)
            solution = [*extracDirectionsFromPath(pathOrder), currentDirection]
            return solution

        for child in problem.getSuccessors(currentPoint):
            childPoint, childDirection, childCost = child
            if childPoint not in explored and child not in frontier.list and childPoint not in path:
                path[childPoint] = currentNude
                frontier.push(child)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
