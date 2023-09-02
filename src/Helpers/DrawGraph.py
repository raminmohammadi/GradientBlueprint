# code for visualisation of each function
from graphviz import Digraph

def trace(root):
    # Builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for children in v._prev:
                edges.add((children, v))
                build(children)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format = 'svg', graph_attr = {'rankdir': "BT"}) # LR = left to right
    nodes, edges = trace(root)

    for n in nodes:
        # Return the identity of an object. This is guaranteed to be unique among simultaneously existing objects. (CPython uses the object's memory address.)
        uid = str(id(n)) 
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | gradient %.10f}" % (n.label, n.data, n.grad),
                 shape='record',  style='filled', fillcolor='#F0F8FF')
        if n._op:
            # If this value is a result of some operations, create an _op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.attr(label=r'Forward Pass')
    dot.attr(fontsize='12')
                 
    return dot