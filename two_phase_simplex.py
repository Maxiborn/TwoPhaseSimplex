import numpy as np
import sys
import math
import random

class TwoPhaseSimplex:
    def _cut_of_cols_from_np_array(self, matrix, start, to, axis):
        for i in range(start, to):
            # 1 => because we want to cut off columns
            # start => defines the index of the column we always cut off (columns are shifted after every deletion)
            matrix = np.delete(matrix, start, axis)
        return matrix

    def _simplex(self, c, b, A, indices):
        """
            Assumption: c, b, A are expected to be NumPy arrays.
        """
        (rows, cols) = A.shape
        max_num_of_iterations = math.factorial(cols)

        while max_num_of_iterations > 0:
            # 1. Step: Split matrix into basis matrix and non-basis matrix
            basis_matrix = self._cut_of_cols_from_np_array(A, rows, cols, 1)
            non_basis_matrix = self._cut_of_cols_from_np_array(A, 0, rows, 1)

            c_basis = self._cut_of_cols_from_np_array(c, rows, cols, 0)
            c_non_basis = self._cut_of_cols_from_np_array(c, 0, rows, 0)

            # Test for optimality
            # 2. Calculate the simplex multiplicators
            basis_matrix_inverse = np.linalg.inv(basis_matrix)
            y_transposed = np.matmul(c_basis, basis_matrix_inverse)

            # 3. Calculate the reduced costs
            y_transposed_N = np.matmul(y_transposed, non_basis_matrix)
            reduced_costs = np.subtract(c_non_basis, y_transposed_N)

            # 4. Examine the optimality
            (reduced_costs_size) = reduced_costs.shape
            if np.all(np.greater_equal(reduced_costs, np.zeros(reduced_costs_size))):
                # Calculate step 8
                b_transposed = b.transpose()
                b_hat = np.matmul(basis_matrix_inverse, b_transposed)

                b_hat_transposed = b_hat.transpose()
                z = np.matmul(c_basis, b_hat_transposed)

                return z, b_hat, reduced_costs, indices, basis_matrix_inverse, c, non_basis_matrix

            # 5. Find the entering variable with the minimizing index
            min_index_from_reduced_costs = rows + np.argmin(reduced_costs)

            # 6. Calculate A_hat_t
            A_t = A[:, [min_index_from_reduced_costs]]
            A_hat_t = np.matmul(basis_matrix_inverse, A_t)

            # 7. Check whether LP is unlimited
            if np.all(np.less_equal(A_hat_t, np.zeros((rows, 1)))):
                print("LP is unlimited.")
                sys.exit(0)

            # 8. Calculate b_hat
            b_transposed = b.transpose()
            b_hat = np.matmul(basis_matrix_inverse, b_transposed)

            b_hat_transposed = b_hat.transpose()
            z = np.matmul(c_basis, b_hat_transposed)

            # 9. Find the leaving variable
            A_hat_t_transposed = A_hat_t.transpose().reshape(len(A_hat_t), )
            min_value_of_leaving_variable = sys.float_info.max
            min_index_of_leaving_variable = -1
            for i in range(rows):
                if A_hat_t_transposed[i] > 0:
                    division = b_hat_transposed[i] / A_hat_t_transposed[i]

                    if division < min_value_of_leaving_variable:
                        min_index_of_leaving_variable = i
                        min_value_of_leaving_variable = division

            # 10. Update: Determine new basis matrix and new basis vector
            temp = A[:, [min_index_of_leaving_variable]]
            A[:, [min_index_of_leaving_variable]] = A[:, [min_index_from_reduced_costs]]
            A[:, [min_index_from_reduced_costs]] = temp

            temp2 = c[min_index_of_leaving_variable]
            c[min_index_of_leaving_variable] = c[min_index_from_reduced_costs]
            c[min_index_from_reduced_costs] = temp2

            temp3 = indices[min_index_of_leaving_variable]
            indices[min_index_of_leaving_variable] = indices[min_index_from_reduced_costs]
            indices[min_index_from_reduced_costs] = temp3

            max_num_of_iterations -= 1

    def _calculate_delta_b(self, inv_basis_matrix, b):
        sensitivity_list = []
        for i in range(len(b)):
            # 1. Construct delta B
            delta_b = np.array([1 if pos == i else 0 for pos in range(len(b))])
            delta_B = np.matmul(inv_basis_matrix, delta_b)

            # 2. Construct B
            B = - np.matmul(inv_basis_matrix, b)

            # 3. Solve the equation
            upper_bound = math.inf # Take the minimum from the division result greater than 0
            lower_bound = -math.inf # Take the maximum from the division result less than 0
            for j in range(len(b)):
                if delta_B[j] != 0:
                    div = B[j] / delta_B[j]

                    if div > 0 and div < upper_bound:
                        upper_bound = div
                    elif div < 0 and div > lower_bound:
                        lower_bound = div

            sensitivity_list.append((lower_bound, upper_bound))

        return sensitivity_list

    def _calculate_delta_c_B(self, inv_basis_matrix, non_basis_matrix, c_basis, c_non_basis):
        sensitivity_list = []
        inv_basis_and_non_basis_product = np.matmul(inv_basis_matrix, non_basis_matrix)
        c_hat_n = c_non_basis - np.matmul(c_basis, inv_basis_and_non_basis_product)

        for i in range(len(c_basis)):
            # Construct the lambda vector.
            lambda_vec = np.zeros(len(c_basis))
            lambda_vec[i] = 1

            # Construct the right side of the equation.
            rhs_equation = np.matmul(lambda_vec, inv_basis_and_non_basis_product)

            # Solve the equation and find the min and max.
            upper_bound = math.inf
            lower_bound = -math.inf
            for j in range(len(rhs_equation)):
                if rhs_equation[j] != 0:
                    div = c_hat_n[j] / rhs_equation[j]

                    if div > 0 and div < upper_bound:
                        upper_bound = div
                    elif div < 0 and div > lower_bound:
                        lower_bound = div

            sensitivity_list.append((lower_bound, upper_bound))

        return sensitivity_list

    def _calculate_shadow_prices(self, z, delta_b, b, basis_matrix_inverse, c_basis):
        shadow_prices = []
        # Construct new b from the ranges of delta b.
        for i, (lower_bound, upper_bound) in enumerate(delta_b):
            if lower_bound == -math.inf and upper_bound == math.inf:
                lower_bound = -1
                upper_bound = 1
            elif lower_bound == -math.inf:
                lower_bound = upper_bound - 10
            elif upper_bound == math.inf:
                upper_bound = lower_bound + 10

            # Get random value for index i.
            lambda_ = random.uniform(lower_bound, upper_bound)
            new_b = np.array(b)
            # Update b.
            new_b[i] += lambda_

            # Calculate the z with the new b value from the sensitivity range.
            new_b_hat = np.matmul(basis_matrix_inverse, new_b.transpose())
            new_z = np.matmul(c_basis, new_b_hat.transpose())

            # Calculate the shadow price.
            factor = (new_z - z) / lambda_ 
            shadow_prices.append(factor)
        
        return shadow_prices


    def two_phase_simplex(self, c, b, A):
        c = np.array(c)
        b = np.array(b)
        A = np.array(A)
        
        indices = [i for i in range(len(A[0]))]

        # 1. Phase
        # 1. Create the new c vector
        # We put the 1's in front for similar reasons as described below for A_extended.
        c_extended = np.array([1 for _ in range(len(b))] + [0 for _ in range(len(c))])
        
        # 2. Extend A with identity matrix
        identity_matrix = np.identity(len(b))
        # We put the identity matrix in front since the simplex algorithm assumes the base matrix
        # is placed in front.
        A_extended = np.concatenate((identity_matrix, A), axis=1)
        # Adapt the indices:
        indices = [i for i in range(len(indices), len(indices) + len(b))] + indices

        # 3. Execute the Simplex algorithm
        z, b_hat, _, indices, _, _, _ = self._simplex(c_extended, b, A_extended, indices)

        # Check whether the first N indices are original indices.
        is_invalid = any(map(lambda x: x >= len(c), indices[:len(b)]))
        if is_invalid:
            print("There is no valid solution for this optimization problem.")
            sys.exit(0)

        # 2. Phase
        # 1. Create a new A while we only keep the columns from the original A.
        updated_A = None
        updated_indices = []
        updated_c = []
        for col_index in indices:
            if col_index < len(A[0]):
                col = A[:, [col_index]]

                if updated_A is None:
                    updated_A = col
                else:
                    updated_A = np.append(updated_A, col, axis=1)
                updated_indices.append(col_index)
                updated_c.append(c[col_index])

        updated_c = np.array(updated_c)

        # 2. Execute the Simplex algorithm for the final result
        z, b_hat, reduced_costs, indices, basis_matrix_inverse, c_sorted, non_basis_matrix = self._simplex(updated_c, b, updated_A, updated_indices)
        delta_b = self._calculate_delta_b(basis_matrix_inverse, b)
        # Prepare c for the sensitivity analysis. Therefore, split it into the basis and non-basis part.
        c_basis = c_sorted[:len(b)]
        c_non_basis = c_sorted[len(b):]
        delta_c_B = self._calculate_delta_c_B(basis_matrix_inverse, non_basis_matrix, c_basis, c_non_basis)

        shadow_prices = self._calculate_shadow_prices(z, delta_b, b, basis_matrix_inverse, c_basis)

        return z, b_hat, indices, delta_b, delta_c_B, reduced_costs, shadow_prices


if __name__ == "__main__":
    """
    c = [-5, -10, 10, 0]
    b = [7, 3]
    A = [[1, 1, -1, 0],
         [1, -2, 2, 1]]

    A = [[105, 3.5, 5, 3.5, 4, 9, 6, 8, 9, 7],
        [0, 103.5, 105, 3.5, 4, 9, 6, 8, 9, 7],
        [0, 0, 0, 103.5, 4, 9, 6, 8, 9, 7],
        [0, 0, 0, 0, 104, 9, 6, 8, 9, 7],
        [0, 0, 0, 0, 0, 109, 106, 8, 9, 7], 
        [0, 0, 0, 0, 0, 0, 0, 108, 9, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 109, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 107]]
    b = [12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000]
    c = [102, 99, 101, 98, 98, 104, 100, 101, 102, 94]


    c = [-5, -10, 10, 0]
    b = [7, 3]
    A = [[1, 1, -1, 0],
         [1, -2, 2, 1]]
    """

    

    # Add input here:
  
    """
    
    c = [103.41, 105.41, 101.95,    #Dedizierte Portfoliostrategie
         107.49, 103.18, 103.20,
         103.39, 102.85, 103.32,
         103.33, 116.98, 99.95,
         117.90, 108.31, 108.74,
         107.87, 107.50, 108.52,
         108.52, 130.24, 107.80,
         108.74, 108.47, 110.09,
         113.41, 112.42, 110.63,
         107.24, 107.54, 137.49,
         139.54, 101.25, 100.29,
         140.56, 140.12, 116.43,
         117.75, 138.69, 137.76,
         120.56
        ]
    b = [6000000, 6000000, 9000000, 9000000,
         10000000, 10000000, 10000000, 10000000,
         8000000, 8000000, 8000000, 8000000,
         6000000, 6000000, 5000000, 5000000]
    A = [[107.875, 108.125, 102.625, 8.125, 2.625, 2.500, 2.250, 1.750, 1.750, 1.625, 7.125, 0.125, 6.250, 2.875, 2.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 108.125, 102.625, 2.500, 2.250, 1.750, 1.750, 1.625, 7.125, 0.125, 6.250, 2.875, 2.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 102.500, 102.250, 101.750, 1.750, 1.625, 7.125, 0.125, 6.250, 2.875, 2.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 101.750, 101.625, 7.125, 0.125, 6.250, 2.875, 2.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107.125, 100.125, 6.250, 2.875, 2.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106.250, 102.875, 102.875,
          2.500, 2.250, 2.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          102.500, 102.250, 102.500, 2.375, 7.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 102.375, 107.500, 2.000, 2.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 102.000, 102.125, 2.000, 2.250, 2.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 102.000, 102.250, 102.875, 2.625, 2.250, 1.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102.625, 102.250, 101.625, 1.625, 6.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101.625, 106.500,
          6.625, 0.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          106.625, 100.625, 0.500, 6.375, 6.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 100.500, 106.375, 106.125, 2.750, 2.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 102.750, 102.875, 5.500, 5.250, 3.125],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 105.500, 105.250, 103.125]]
    
    
    
    c = [-1, -1, 0, 0]
    b = [4, 6]
    A = [[1, 0.5, 1, 0],    #Einführungsbeispiel graphisch
         [0.5, 1, 0, 1]]
    
   
    c = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    b = [14, 13, 15, 16, 19, 18, 11]
    A = [[1, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 1, 0, -1, 0, 0, 0, 0, 0],  #Einführungsbsp. Personalmanagement
         [1, 1, 1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1]]
    

    c = [-0.5, -3, 0, 0, 0]
    b = [6, 2, 8]                       #unbeschränktes LP 
    A = [[0, 1, 1, 0, 0],
         [1, 0, 0, -1, 0],
         [1, 2, 0, 0, -1]]
     
    c = [-2.0000000000000001, -1, 0, 0]
    b = [1, 3]                       #falsch, weil Rundungsfehler 
    A = [[0, 1, 1, 0],
         [2, 1, 0, 1]]
    
    c = [-2.000000000000001, -1, 0, 0]
    b = [1, 3]                       #richtig 
    A = [[0, 1, 1, 0],
         [2, 1, 0, 1]]
     
     
    

    c = [2, 5, 0, 0, 0] 
    b = [4, 5, 8]                       #keine zul. Lösung 
    A = [[1, 0, -1, 0, 0],
         [0, 1, 0, -1, 0],
         [1, 2, 0, 0, 1]]

    """

    c = [-1.0, -1.0, 0.0, 0.0, 0.0] 
    b = [5.0, 1.0, 4.0]                       #Beispiel sensi 
    A = [[0.5, 1.0, 1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 1.0]]

    algo = TwoPhaseSimplex()
    z, b_hat, indices, delta_b, delta_c_B, reduced_costs, shadow_prices = algo.two_phase_simplex(c, b, A)
    indices_ = np.array(indices) + 1

    print("Results:")
    print("z: {}".format(z))
    print("Indices: {}".format(str(indices_)))
    print("x_B: {}".format(str(b_hat)))

    print("Sensitivity:")
    print("Reduced costs: {}".format(str(reduced_costs)))
    print("Delta c: {}".format(delta_c_B))
    print("Delta b: {}".format(delta_b))
    print("Shadow prices: {}".format(shadow_prices))

"""
    How to run the code:
        1. Go to the Terminal
        2. cd Desktop
        3. python3 two_phase_simplex.py
"""
