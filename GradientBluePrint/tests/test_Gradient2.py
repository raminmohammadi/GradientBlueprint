import unittest
import sys, os

# # Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# # Add the parent directory to the Python path
grandparent_directory = os.path.dirname(parent_directory)

sys.path.append(grandparent_directory)

from src.Gradient.Gradient import Variable


class Test_Gradient(unittest.TestCase):
    """Tests the following functions of Variable class.
    6. __truediv__()
    7. __neg__()
    8. __rsub__()
    9. __radd__()
    10. __rtruediv__()
    11. __rmul__()
    
    """
    def setUp(self):
        """This method recreates variables for each new test."""
        
        self.a = Variable(0)
        self.b = Variable(1.0); self.b.label = 'b'
        self.c = Variable(-2.0); self.c.label = 'c'

    def validate_variable_attributes(self, variable, data, prev, _op, label, grad):
        """Validates each attribute of variable."""
        
        self.assertEqual(variable.data, data)
        #self.assertEqual(variable._prev, prev)
        self.assertEqual( {child.data for child in variable._prev}, {child.data for child in prev})
        self.assertEqual(variable._op, _op)
        self.assertEqual(variable.label, label)
        self.assertEqual(variable.grad, grad)
        self.assertEqual(variable._backward(), None)

    #------------------------------TESTS------------------------------
    def test_variable(self):
        """Verifies the attributes of newly created `Variable` instance."""

        self.validate_variable_attributes(self.a, 0, set(), '', '', 0.0)
        self.validate_variable_attributes(self.b, 1.0, set(), '', 'b', 0.0)
        self.validate_variable_attributes(self.c, -2.0, set(), '', 'c', 0.0)

    def test_truediv(self):
        """Tests __truediv__() function"""

         # a / a
        with self.assertRaises(AssertionError):
            result = self.a / self.a
        
        # a / b
        result = self.a / self.b
        self.validate_variable_attributes(result, 0.0, {self.a, self.b**-1}, '*', '', 0.0)

        # a / c
        result = self.a / self.c
        self.validate_variable_attributes(result, 0.0, {self.a, self.c**-1}, '*', '', 0.0)

        # b / a
        with self.assertRaises(AssertionError):
            result = self.b / self.a

        # b / b
        result = self.b / self.b
        self.validate_variable_attributes(result, 1.0, {self.b, self.b**-1}, '*', '', 0.0)

        # b / c
        result = self.b / self.c
        self.validate_variable_attributes(result, -0.5, {self.b, self.c**-1}, '*', '', 0.0)
        
        # c / a
        with self.assertRaises(AssertionError):
            result = self.c / self.a

        # c / b
        result = self.c/self.b
        self.validate_variable_attributes(result, -2.0, {self.c, self.b**-1}, '*', '', 0.0)
        
        # c / c
        result = self.c / self.c
        self.validate_variable_attributes(result, 1.0, {self.c, self.c**-1}, '*', '', 0.0)
    
    def test_neg(self):
        """Tests __neg__() function"""

        # -a
        result = -self.a
        self.validate_variable_attributes(result, 0.0, {Variable(-1), self.a}, '*', '', 0.0)

        # -b
        result = -self.b
        self.validate_variable_attributes(result, -1.0, {Variable(-1), self.b}, '*', '', 0.0)
        
        # -c
        result = -self.c
        self.validate_variable_attributes(result, 2.0, {Variable(-1), self.c}, '*', '', 0.0)

    def test_rsub(self):
        """Tests __rsub__() function"""

        # 0-0
        result = 0-self.a
        self.validate_variable_attributes(result, 0.0, {Variable(0), self.a}, '-', '', 0.0)

        # 5-b
        result = 5-self.b
        self.validate_variable_attributes(result, 4.0, {Variable(5), self.b}, '-', '', 0.0)
        
        # 10-c
        result = 10-self.c
        self.validate_variable_attributes(result, 12.0, {Variable(10), self.c}, '-', '', 0.0)

    def test_radd(self):
        """Tests __radd__() function"""
        
        # 0+0
        result = 0+self.a
        self.validate_variable_attributes(result, 0.0, {Variable(0), self.a}, '+', '', 0.0)

        # 5+b
        result = 5+self.b
        self.validate_variable_attributes(result, 6.0, {Variable(5), self.b}, '+', '', 0.0)
        
        # 10+c
        result = 10+self.c
        self.validate_variable_attributes(result, 8.0, {Variable(10), self.c}, '+', '', 0.0)

    def test_rmul(self):
        """Tests __rmul__() function"""
        
        # 0*a = 0
        result = 0*self.a
        self.validate_variable_attributes(result, 0.0, {Variable(0), self.a}, '*', '', 0.0)

        # 5*b = 5.0
        result = 5*self.b
        self.validate_variable_attributes(result, 5.0, {Variable(5), self.b}, '*', '', 0.0)
        
        # 10*c = -20.0
        result = 10*self.c
        self.validate_variable_attributes(result, -20.0, {Variable(10), self.c}, '*', '', 0.0)

    def test_rtruediv(self):
        """Tests __rtruediv__() function"""
        
        # 0/a = 0
        with self.assertRaises(AssertionError):
            result = 0/self.a

        # 5/b = 5.0
        result = 5/self.b
        self.validate_variable_attributes(result, 5.0, {Variable(5), self.b**-1}, '*', '', 0.0)
        
        # 10/c = -5.0
        result = 10/self.c
        self.validate_variable_attributes(result, -5.0, {Variable(10), self.c**-1}, '*', '', 0.0)
        
if __name__ == '__main__':
    unittest.main()