# ----------------------------------------------------------------------------------------------------------------------
# Index:0, Title: Maximum Subarray, [Dynamic programming]
# ----------------------------------------------------------------------------------------------------------------------
# Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum
# and return its sum.
#
# Example:
#
# Input: [-2,1,-3,4,-1,2,1,-5,4],
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.
# Follow up:
#
# If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which
# is more subtle.
# ----------------------------------------------------------------------------------------------------------------------
class Solution:
    # @param A, a list of integers
    # @return an integer
    # 6:57
    def maxSubArray(self, A):
        if not A:
            return 0

        curSum = maxSum = A[0]
        for num in A[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)

        return maxSum
# ----------------------------------------------------------------------------------------------------------------------
# Index:1, Title: Unique Paths, [Dynamic programming]
# ----------------------------------------------------------------------------------------------------------------------
# A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
#
# The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right
#  corner of the grid (marked 'Finish' in the diagram below).
#
# How many possible unique paths are there?
#
# Above is a 7 x 3 grid. How many possible unique paths are there?
#
# Note: m and n will be at most 100.
# ----------------------------------------------------------------------------------------------------------------------
# math C(m+n-2,n-1)
import math
def uniquePaths1(self, m, n):
    if not m or not n:
        return 0
    return math.factorial(m + n - 2) / (math.factorial(n - 1) * math.factorial(m - 1))


# O(m*n) space   
def uniquePaths2(self, m, n):
    if not m or not n:
        return 0
    dp = [[1 for _ in range(n)] for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


# O(n) space 
def uniquePaths(self, m, n):
    if not m or not n:
        return 0
    cur = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            cur[j] += cur[j - 1]
    return cur[-1]
# ----------------------------------------------------------------------------------------------------------------------
# Index:3, Title: Climbing Stairs, [Dynamic programming]
# ----------------------------------------------------------------------------------------------------------------------
# You are climbing a stair case. It takes n steps to reach to the top.
#
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
#
# Note: Given n will be a positive integer.
#
# Example 1:
#
# Input: 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
# Example 2:
#
# Input: 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
# ----------------------------------------------------------------------------------------------------------------------
# Top down + memorization (dictionary)
def __init__(self):
    self.dic = {1: 1, 2: 2}


def climbStairs(self, n):
    if n not in self.dic:
        self.dic[n] = self.climbStairs(n - 1) + self.climbStairs(n - 2)
    return self.dic[n]

# ----------------------------------------------------------------------------------------------------------------------
# Index:3, Title: Climbing Stairs, [Dynamic programming]
# ----------------------------------------------------------------------------------------------------------------------
# You are given coins of different denominations and a total amount of money amount. Write a function to compute
#  the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any
#  combination of the coins, return -1.
#
# Example 1:
# coins = [1, 2, 5], amount = 11
# return 3 (11 = 5 + 5 + 1)
#
# Example 2:
# coins = [2], amount = 3
# return -1.
#
# Note:
# You may assume that you have an infinite number of each kind of coin.
# ----------------------------------------------------------------------------------------------------------------------
#Assume dp[i] is the fewest number of coins making up amount i,
# then for every coin in coins, dp[i] = min(dp[i - coin] + 1).
#The time complexity is O(amount * coins.length) and the space complexity is O(amount)

class Solution(object):
    def coinChange(self, coins, amount):
        MAX = float('inf')
        dp = [0] + [MAX] * amount

        for i in range(1, amount + 1):
            dp[i] = min([dp[i - c] if i - c >= 0 else MAX for c in coins]) + 1

        return [dp[amount], -1][dp[amount] == MAX]
# ----------------------------------------------------------------------------------------------------------------------
# Index:4, Title: Course Schedule, [DFS ]
# ----------------------------------------------------------------------------------------------------------------------
# There are a total of n courses you have to take, labeled from 0 to n-1.
#
# Some courses may have prerequisites, for example to take course 0 you have to first take course 1,
# which is expressed as a pair: [0,1]
#
# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
#
# Example 1:
#
# Input: 2, [[1,0]]
# Output: true
# Explanation: There are a total of 2 courses to take.
#              To take course 1 you should have finished course 0. So it is possible.
# Example 2:
#
# Input: 2, [[1,0],[0,1]]
# Output: false
# Explanation: There are a total of 2 courses to take.
#              To take course 1 you should have finished course 0, and to take course 0 you should
#              also have finished course 1. So it is impossible.
# Note:
#
# The input prerequisites is a graph represented by a list of edges, not adjacency matrices.
# Read more about how a graph is represented.
# You may assume that there are no duplicate edges in the input prerequisites.
#  Hints:
#
# This problem is equivalent to finding if a cycle exists in a directed graph.
# If a cycle exists, no topological ordering exists and therefore it will be impossible to take all courses.
# Topological Sort via DFS -
# A great video tutorial (21 minutes) on Coursera explaining the basic concepts of Topological Sort.
# Topological sort could also be done via BFS.
# ----------------------------------------------------------------------------------------------------------------------
def canFinish(self, numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    visit = [0 for _ in range(numCourses)]
    for x, y in prerequisites:
        graph[x].append(y)
    def dfs(i):
        if visit[i] == -1:
            return False
        if visit[i] == 1:
            return True
        visit[i] = -1
        for j in graph[i]:
            if not dfs(j):
                return False
        visit[i] = 1
        return True
    for i in range(numCourses):
        if not dfs(i):
            return False
    return True

# if node v has not been visited, then mark it as 0.
# if node v is being visited, then mark it as -1. If we find a vertex marked as -1 in DFS, then their is a ring.
# if node v has been visited, then mark it as 1. If a vertex was marked as 1, then no ring contains v or its successors.
# ----------------------------------------------------------------------------------------------------------------------
# Index:5, Title: Course Schedule, [DFS ]
# ----------------------------------------------------------------------------------------------------------------------