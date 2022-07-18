from tkinter import Y
import numpy as np 

def matrix_multiplication(A, B):
    ## n x n matricies
    n = len(A)
    C = [[0] * n] * n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C



A = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
B = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
print(matrix_multiplication(A, B))

def split(matrix): 
    """ 
    Splits a given matrix into quarters. 
    Input: nxn matrix 
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d 
    """
    row, col = matrix.shape 
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:] 
  



def strassen(x, y): 
    """ 
    Computes matrix product by divide and conquer approach, recursively. 
    Input: nxn matrices x and y 
    Output: nxn matrix, product of x and y 
    """
  
    # Base case when size of matrices is 1x1 
    if len(x) == 1: 
        return x * y 
  
    # Splitting the matrices into quadrants. This will be done recursively 
    # until the base case is reached. 
    a, b, c, d = split(x) 
    e, f, g, h = split(y) 
  
    # Computing the 7 products, recursively (p1, p2...p7) 
    p1 = strassen(a, f - h)   
    p2 = strassen(a + b, h)         
    p3 = strassen(c + d, e)         
    p4 = strassen(d, g - e)         
    p5 = strassen(a + d, e + h)         
    p6 = strassen(b - d, g + h)   
    p7 = strassen(a - c, e + f)   
  
    # Computing the values of the 4 quadrants of the final matrix c 
    c11 = p5 + p4 - p2 + p6   
    c12 = p1 + p2            
    c21 = p3 + p4             
    c22 = p1 + p5 - p3 - p7   
  
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically. 
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))  
  
    return c 
A = np.array(A)
B = np.array(B)
#print(strassen(A, B))



def new_matrix(r, c):
    """Create a new matrix filled with zeros."""
    matrix = [[0 for row in range(r)] for col in range(c)]
    return matrix


def direct_multiply(x, y):
    if len(x[0]) != len(y):
        return "Multiplication is not possible!"
    else:
        p_matrix = new_matrix(len(x), len(y[0]))
        for i in range(len(x)):
            for j in range(len(y[0])):
                for k in range(len(y)):
                    p_matrix[i][j] += x[i][k] * y[k][j]

    return p_matrix


def split(matrix):
    """Split matrix into quarters."""
    a = b = c = d = matrix

    while len(a) > len(matrix)/2:
        a = a[:len(a)//2]
        b = b[:len(b)//2]
        c = c[len(c)//2:]
        d = d[len(d)//2:]

    while len(a[0]) > len(matrix[0])//2:
        for i in range(len(a[0])//2):
            a[i] = a[i][:len(a[i])//2]
            b[i] = b[i][len(b[i])//2:]
            c[i] = c[i][:len(c[i])//2]
            d[i] = d[i][len(d[i])//2:]

    return a, b, c, d


def add_matrix(a, b):
    if type(a) == int:
        d = a + b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] + b[i][j])
            d.append(c)
    return d


def subtract_matrix(a, b):
    if type(a) == int:
        d = a - b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] - b[i][j])
            d.append(c)
    return d


def strassen(x, y, n):
    # base case: 1x1 matrix
    if n == 1:
        z = [[0]]
        z[0][0] = x[0][0] * y[0][0]
        return z
    else:
        # split matrices into quarters
        a, b, c, d = split(x)
        e, f, g, h = split(y)

        # p1 = a*(f-h)
        p1 = strassen(a, subtract_matrix(f, h), n/2)

        # p2 = (a+b)*h
        p2 = strassen(add_matrix(a, b), h, n/2)

        # p3 = (c+d)*e
        p3 = strassen(add_matrix(c, d), e, n/2)

        # p4 = d*(g-e)
        p4 = strassen(d, subtract_matrix(g, e), n/2)

        # p5 = (a+d)*(e+h)
        p5 = strassen(add_matrix(a, d), add_matrix(e, h), n/2)

        # p6 = (b-d)*(g+h)
        p6 = strassen(subtract_matrix(b, d), add_matrix(g, h), n/2)

        # p7 = (a-c)*(e+f)
        p7 = strassen(subtract_matrix(a, c), add_matrix(e, f), n/2)

        z11 = add_matrix(subtract_matrix(add_matrix(p5, p4), p2), p6)

        z12 = add_matrix(p1, p2)

        z21 = add_matrix(p3, p4)

        z22 = add_matrix(subtract_matrix(subtract_matrix(p5, p3), p7), p1)

        z = new_matrix(len(z11)*2, len(z11)*2)
        for i in range(len(z11)):
            for j in range(len(z11)):
                z[i][j] = z11[i][j]
                z[i][j+len(z11)] = z12[i][j]
                z[i+len(z11)][j] = z21[i][j]
                z[i+len(z11)][j+len(z11)] = z22[i][j]

        return z

###################################################
def direct_multiply(x, y):
    if len(x[0]) != len(y):
        return "Multiplication is not possible!"
    else:
        p_matrix = np.zeros((len(x), len(y[0])))
        for i in range(len(x)):
            for j in range(len(y[0])):
                for k in range(len(y)):
                    p_matrix[i][j] += x[i][k] * y[k][j]

    return p_matrix


def np_split(matrix):
    """Split matrix into quarters."""
    matrix = np.array(matrix)

    row, col = matrix.shape 
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]




def add_matrix(a, b):
    if type(a) == int:
        d = a + b
    else:
        a, b = np.array(a), np.array(b)
        d = a + b
    return d


def subtract_matrix(a, b):
    if type(a) == int:
        d = a - b
    else:
        a, b = np.array(a), np.array(b)
        d = a - b
    return d


def strassen(x, y, n):
    # base case: 1x1 matrix
    if n == 1:
        z = [[0]]
        z[0][0] = x[0][0] * y[0][0]
        return z
    else:
        # split matrices into quarters
        a, b, c, d = np_split(x)
        e, f, g, h = np_split(y)

        # p1 = a*(f-h)
        p1 = strassen(a, subtract_matrix(f, h), n/2)

        # p2 = (a+b)*h
        p2 = strassen(add_matrix(a, b), h, n/2)

        # p3 = (c+d)*e
        p3 = strassen(add_matrix(c, d), e, n/2)

        # p4 = d*(g-e)
        p4 = strassen(d, subtract_matrix(g, e), n/2)

        # p5 = (a+d)*(e+h)
        p5 = strassen(add_matrix(a, d), add_matrix(e, h), n/2)

        # p6 = (b-d)*(g+h)
        p6 = strassen(subtract_matrix(b, d), add_matrix(g, h), n/2)

        # p7 = (a-c)*(e+f)
        p7 = strassen(subtract_matrix(a, c), add_matrix(e, f), n/2)

        z11 = add_matrix(subtract_matrix(add_matrix(p5, p4), p2), p6)

        z12 = add_matrix(p1, p2)

        z21 = add_matrix(p3, p4)

        z22 = add_matrix(subtract_matrix(subtract_matrix(p5, p3), p7), p1)

        z = np.zeros((len(z11)*2, len(z11)*2))
        for i in range(len(z11)):
            for j in range(len(z11)):
                z[i][j] = z11[i][j]
                z[i][j+len(z11)] = z12[i][j]
                z[i+len(z11)][j] = z21[i][j]
                z[i+len(z11)][j+len(z11)] = z22[i][j]

        return z




a = np.array([[11,11,11,11],[22,22,22,22],[33,33,33,33],[44,44,44,44]])
b = np.array([[101,181,119,113],[22,22,22,22],[33,33,33,33],[44,44,44,44]])

print(f"a = {a}")
print(f"b = {b}")

print(f"Using Strassen's algorithm:\na*b = {strassen(a, b, 4)}")

print(f"Using naive algorithm:\na*b = {direct_multiply(a, b)}")


