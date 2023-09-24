import math


class Variable:
    def __init__(self, data, _children=(), _op = '', label = ''):
        """
        @ data: float  # input data
        @ _children: set, To keep track of all connections, basically to keep track of what variables are producing what Variables
        @ _op: str, To keep track of mathematical expression of each operation
        """
        
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0 # derivative of the value with respect to an _childern
        self._backward = lambda: None # this is responsible for the chain-rule value - a function that by default does nothing
        
    def __add__(self, other):
        """
        To add two Variable objects
        """
        other = other if isinstance(other, Variable) else Variable(other) 
        output = Variable(self.data + other.data, _children=(self, other), _op = '+')

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
            
        output._backward = _backward
        return output

    def __sub__(self, other):
        """
        To subtract two Variable objects
        """
        other = other if isinstance(other, Variable) else Variable(other) 
        output = Variable(self.data - other.data, _children=(self, other), _op = '-')

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
            
        output._backward = _backward
        return output
        
    def __mul__(self, other):
        """
        To multiply two Variable objects
        """
        other = other if isinstance(other, Variable) else Variable(other) 
        output = Variable(self.data * other.data, _children=(self, other), _op = '*')
        
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
            
        output._backward = _backward
        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        assert (self.data != 0.0 or (self.data ==0.0 and other > 0.0)), "0.0 can only be raised to a positive power"
        output = Variable(self.data ** other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * output.grad
            
        output._backward = _backward
        return output            
            

    def exp(self):
        x = self.data
        output = Variable(math.exp(x), _children=(self, ), _op='exp')

        def _backward():
            self.grad += output.grad * output.data
        output._backward = _backward
        return output
            

    def backward(self):
        topological_graph = []
        visited_nodes = set()
        
        def build_topological_graph(v):
            if v not in visited_nodes:
                visited_nodes.add(v)
                for child in v._prev:
                    build_topological_graph(child)
                topological_graph.append(v)
        
        build_topological_graph(self)

        self.grad = 1.0
        for node in reversed(topological_graph):
            node._backward()

    def __draw__(self):
        return draw_dot(self)
                
    def __repr__(self):
        """
        __repr__ is a built-in method in Python that returns a string representation of an object.
        """
        return f"Variable(data={self.data})"

    def __truediv__(self, other):
        """
        Return a / b where 2/3 is .66 rather than 0. This is also known as “true” division.
        """
        assert (other.data!=0), "Division by 0 is undefined"
        
        return self * other**-1

    def __neg__(self):
        """
        Return obj negated (-obj).
        """
        return self * -1

    def __rsub__(self, other):
        """
         __rsub__() method implements the reverse subtraction operation that is subtraction with reflected,
         swapped operands. # other - self
        """
        #return other - self
        return Variable(other) - self

    def __radd__(self, other): # other + self
        return self + other

    def __rtruediv__(self, other): # other / self
        """
        The Python __rtruediv__() method implements the reverse true division operation with reflected, 
        swapped operands. So, when you call other / self,
        """
        return other * self**-1

    def __rmul__(self, other):
        """
        Python can not multiply other * self if other is simply a digit. Ex: 2 * Variable(4)
        so we define __rmul__ which by default will run if __mul__ operation was not possible. 
        """
        return self * other

    # def __eq__(self, other):
    #     return self.data == other.data