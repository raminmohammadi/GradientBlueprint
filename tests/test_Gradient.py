import unittest
import math
import sys

sys.path.append('../src/Gradient/')

from Gradient import Variable

class Test_Gradient(unittest.TestCase):

    def setUp(self):
        """This method recreates variables for each new test"""
        
        self.a = Variable(0)
        self.b = Variable(1.0); self.b.label = 'b'
        self.c = Variable(-2.0); self.c.label = 'c'

    def validate_variable_attributes(self, variable, data, prev, _op, label, grad):
        """Validates each attribute of variable"""
        
        self.assertEqual(variable.data, data)
        self.assertEqual(variable._prev, prev)
        self.assertEqual(variable._op, _op)
        self.assertEqual(variable.label, label)
        self.assertEqual(variable.grad, grad)
        self.assertEqual(variable._backward(), None)

    #------------------------------TESTS------------------------------
    def test_variable(self):
        """Verifies the attributes of newly created `Variable` instance"""

        self.validate_variable_attributes(self.a, 0, set(), '', '', 0.0)
        self.validate_variable_attributes(self.b, 1.0, set(), '', 'b', 0.0)
        self.validate_variable_attributes(self.c, -2.0, set(), '', 'c', 0.0)
    
    def test_add(self):
        """Tests add fn"""

        # a + a
        result = self.a + self.a
        self.validate_variable_attributes(result, 0, {self.a}, '+', '', 0.0)

        # a + b
        result = self.a + self.b
        self.validate_variable_attributes(result, 1, {self.a, self.b}, '+', '', 0.0)

        # b + c
        result = self.b + self.c
        self.validate_variable_attributes(result, -1.0, {self.b, self.c}, '+', '', 0.0)

        # c + a
        result = self.c + self.a
        self.validate_variable_attributes(result, -2.0, {self.c, self.a}, '+', '', 0.0)

    def test_subract(self):
        """Tests subtract fn"""
        
        # a - a
        result = self.a - self.a
        self.validate_variable_attributes(result, 0, {self.a}, '-', '', 0.0)

        # a - b
        result = self.a - self.b
        self.validate_variable_attributes(result, -1.0, {self.a, self.b}, '-', '', 0.0)

        # b - c
        result = self.b - self.c
        self.validate_variable_attributes(result, 3.0, {self.b, self.c}, '-', '', 0.0)

        # c - a
        result = self.c - self.a
        self.validate_variable_attributes(result, -2.0, {self.c, self.a}, '-', '', 0.0)
    
    def test_mulitply(self):
        """Tests multiply fn"""
    
         # a * a
        result = self.a * self.a
        self.validate_variable_attributes(result, 0, {self.a}, '*', '', 0.0)
        
        # a * b
        result = self.a * self.b
        self.validate_variable_attributes(result, 0.0, {self.a, self.b}, '*', '', 0.0)
        
        # b * c
        result = self.b * self.c
        self.validate_variable_attributes(result, -2.0, {self.b, self.c}, '*', '', 0.0)
        
        # c * a
        result = self.c * self.a
        self.validate_variable_attributes(result, 0.0, {self.c, self.a}, '*', '', 0.0)   
        
        # c * c
        result = self.c * self.c
        self.validate_variable_attributes(result, 4.0, {self.c}, '*', '', 0.0)   

    def test_pow(self):
        """Tests power fn"""

        # a ** 0
        with self.assertRaises(AssertionError):
            result = self.a ** 0
        
        # a ** 1.0
        result = self.a ** 1.0
        self.validate_variable_attributes(result, 0.0, {self.a}, '**1.0', '', 0.0)
        
        # b ** -2.0
        result = self.b ** -2.0
        self.validate_variable_attributes(result, 1.0, {self.b}, '**-2.0', '', 0.0)
        
        # c ** 0
        result = self.c ** 0
        self.validate_variable_attributes(result, 1.0, {self.c}, '**0', '', 0.0)   
        
        # c ** -2.0
        result = self.c ** -2.0
        self.validate_variable_attributes(result, 0.25, {self.c}, '**-2.0', '', 0.0)

    def test_exp(self):
        """Tests exp fn"""

         # e * 0.0
        result = self.a.exp()
        self.validate_variable_attributes(result, 1.0, {self.a}, 'exp', '', 0.0)
        
        # e * 1.0
        result = self.b.exp()
        self.validate_variable_attributes(result, math.exp(1.0), {self.b}, 'exp', '', 0.0)
        
        # e * -2.0
        result = self.c.exp()
        self.validate_variable_attributes(result, math.exp(-2.0), {self.c}, 'exp', '', 0.0)
        
if __name__ == '__main__':
    unittest.main()