//
// Created by korone on 1/13/25.
//

#include "TopologicalSort.h"
#include <queue>
#include <unordered_map>

std::vector<int> TopologicalSort::topologicalSort(const std::unordered_map<int, Node> &nodes,
                                      const std::vector<Connection> &connections)
{
    // 1) Build adjacency list and in-degree map using node IDs
    std::unordered_map<int, int> inDegree;
    std::unordered_map<int, std::vector<int>> adjacencyList;

    // Initialize all in-degrees to 0
    for (const auto &[id, node] : nodes)
    {
        inDegree[id] = 0;
    }

    // Populate adjacency list and in-degree counts
    for (const auto &conn : connections)
    {
        if (conn.enabled)
        {
            adjacencyList[conn.inputNode].push_back(conn.outputNode);
            inDegree[conn.outputNode]++;
        }
    }

    // 2) Collect nodes with zero in-degree
    std::queue<int> zeroInDegreeQueue;
    for (const auto &pair : inDegree)
    {
        if (pair.second == 0)
        {
            zeroInDegreeQueue.push(pair.first); // node ID
        }
    }

    // 3) Perform topological sort on IDs
    std::vector<int> topoOrder;
    topoOrder.reserve(nodes.size());

    while (!zeroInDegreeQueue.empty())
    {
        int currentId = zeroInDegreeQueue.front();
        zeroInDegreeQueue.pop();
        topoOrder.push_back(currentId);

        // Decrement in-degree of neighbors
        if (adjacencyList.count(currentId))
        {
            for (int neighborId : adjacencyList[currentId])
            {
                inDegree[neighborId]--;
                if (inDegree[neighborId] == 0)
                {
                    zeroInDegreeQueue.push(neighborId);
                }
            }
        }
    }

    std::vector<int> result;
    result.reserve(topoOrder.size());
    for (const auto id : topoOrder)
    {
        result.push_back(id);
    }

    return result;
}
