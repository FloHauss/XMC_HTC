import copy
# puts all leaf paths in the leaf_nodes list (each represented as a tree)
def find_leaf_trees(root, tree):

    leaf_trees = []

    def helper(node, current_tree, leaf_trees):
        # Base Case: Leaf Node
        if node not in tree:
            leaf_trees.append(dict(current_tree))
        # Else: repeat the process for each child
        else:
            current_tree[node] = []
            for child in tree[node]:
                new_tree = copy.deepcopy(current_tree) # deepcopy required!
                new_tree[node].append(child)
                helper(child, new_tree, leaf_trees)

    helper(root, {}, leaf_trees)
    return leaf_trees

def tree_merge(tree_1, tree_2):
    merger = copy.deepcopy(tree_1)
    for k, v in tree_2.items():
        if k in merger.keys():
            merger[k] = list(set(merger[k] + v))
        else:
            merger[k] = v
    return merger

# Returns all the nodes of a tree in a list
def flatten_tree(tree):
    nodes = set()
    for k, v in tree.items():
        nodes.add(k)
        for child in v:
            nodes.add(child)
    return nodes

# Merges trees in the list from left to right such that the resulting trees have at maximum k elements
def k_merger(k, sub_trees):
    merged_trees = []
    current_tree = {}
    while sub_trees:
        merged_tree = tree_merge(current_tree, sub_trees[0])
        if len(flatten_tree(merged_tree)) <= k:
            current_tree = merged_tree
            sub_trees.pop(0)
        else:
            merged_trees.append(current_tree)
            current_tree = {}
    if current_tree:
        merged_trees.append(current_tree)
    return merged_trees