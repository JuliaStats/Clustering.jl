# Union-Find
# structure for managing disjoint sets
# This structure tracks which sets the elements of a set belong to,
# and allows us to efficiently determine whether two elements belong to the same set.
mutable struct UnionFind
    parent::Vector{Int}  # parent[root] is the negative of the size
    label::Dict{Int, Int}
    next_id::Int

    function UnionFind(nodes::Int)
        if nodes <= 0
            throw(ArgumentError("invalid argument for nodes: $nodes"))
        end

        parent = -ones(nodes)
        label = Dict([i=>i for i in 1:nodes])
        new(parent, label, nodes)
    end
end

# label of the set which element `x` belong to
set_id(uf::UnionFind, x::Int) = uf.label[root(uf, x)]
# all elements that have the specified label
items(uf::UnionFind, x::Int) = [k for (k, v) in pairs(uf.label) if v == x]

# root of element `x`
# The root is the representative element of the set
function root(uf::UnionFind, x::Int)
    if uf.parent[x] < 0
        return x
    else
        return uf.parent[x] = root(uf, uf.parent[x])
    end
end

function setsize(uf::UnionFind, x::Int)
    return -uf.parent[root(uf, x)]
end

function unite!(uf::UnionFind, x::Int, y::Int)
    x = root(uf, x)
    y = root(uf, y)
    if x == y
        return false
    end
    if uf.parent[x] > uf.parent[y]
        x, y = y, x
    end
    # unite smaller tree(y) to bigger one(x)
    uf.parent[x] += uf.parent[y]
    uf.parent[y] = x
    uf.next_id += 1
    uf.label[y] = uf.next_id
    for i in items(uf, set_id(uf, x))
        uf.label[i] = uf.next_id
    end
    return true
end