import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, initial_guess, iterations=20):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.
    :param a: matrix A
    :param b: vector b
    :param initial_guess: Initial guess for the solution 𝑿0
    :param iterations: Number of iterations
    :return: Last iteration solution and maximum of the difference array
    """
    X = initial_guess.copy()  # 𝑿𝒌
    n = len(b)
    max_diff_array = np.zeros(iterations)  # vector d

    for k in range(iterations):
        x_old = X.copy()
        for i in range(n):
            sum_except_i = np.dot(A[i, :], X) - A[i, i] * X[i]
            X[i] = (b[i] - sum_except_i) / A[i, i]

            # Capture the maximum absolute difference
            max_diff_array[k] = np.max(np.abs(X - x_old))

    return X, max_diff_array


# Inputs for matrix A and vector b
# for linear equation1
"""
matrix_a = np.array([[4, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 4]], dtype=float)

vector_b = np.array([15, 10, 10], dtype=float)
initial_guess = np.array([0, 0, 0], dtype=float)  # initial vector 
"""
# for linear equation2
matrix_a = np.array([[1, 4, 1],
                     [2, 1, 3],
                     [4, 2, 1],
                     ], dtype=float)
vector_b = np.array([4, 7, 5], dtype=float)
initial_guess = np.array([2, 2, 2], dtype=float)  # initial vector 𝑿𝟎
# Perform Gauss-Seidel iteration
solution, max_diff = gauss_seidel(matrix_a, vector_b, initial_guess)
print("Solution vector X:")
print(solution)
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(max_diff, linestyle='-', color='b')
plt.xlabel('Iteration Index (k)')
plt.ylabel('Max Difference |X_k - X_(k-1)|')
plt.grid(True)
plt.show()
...
