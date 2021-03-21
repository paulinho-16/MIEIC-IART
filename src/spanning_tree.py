
# Kruskal's algorithm in Python
def kruskal(num_routers, edges):
    edges = sorted(edges, key=lambda edge: edge[2])

    vertices = list(range(num_routers))

    def find_parent(i):
        if i != vertices[i]:
            vertices[i] = find_parent(vertices[i])
        return vertices[i]

    minimum_spanning_tree_cost = 0
    minimum_spanning_tree = []

    for edge in edges:
        parent_a = find_parent(edge[0])
        parent_b = find_parent(edge[1])
        if parent_a != parent_b:
            minimum_spanning_tree_cost += edge[2]
            minimum_spanning_tree.append(edge)
            vertices[parent_a] = parent_b

    return (minimum_spanning_tree_cost,minimum_spanning_tree)
