from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]      #Example Personalmanagement
    b = [14, 13, 15, 16, 19, 18, 11]
    A = [[1, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 1, 0, -1, 0, 0, 0, 0, 0],  
         [1, 1, 1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1]]

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
    Output:



    Results:
    
    z: 22.0
    Indices: [ 4  1  3  2  5  9  6  7  8 10 11 12 13 14]
    x_B: [4. 4. 1. 7. 3. 4. 3.]

    Sensitivity:

    Reduced costs: [0.33333333 0.33333333 0.33333333 0.
                    0.33333333 0.33333333 0.        ]
    Delta c: [(-1.0, 0.49999999999999983), (-0.9999999999999998, 0.5000000000000001),
              (-0.9999999999999997, 0.49999999999999994), (-0.3333333333333333, 6004799503160660.0),
              (-0.33333333333333326, inf), (-0.20000000000000004, inf), (-1.0, 0.4999999999999999)]
    Delta b: [(-6.000000000000004, 1.4999999999999958), (-inf, 4.0000000000000036),
              (-2.9999999999999925, 5.9999999999999964), (-3.9999999999999973, 3.0),
              (-2.9999999999999916, 4.5), (-6.000000000000004, 1.4999999999999958),
              (-0.9999999999999973, 4.0000000000000036)]
    Shadow prices: [0.0, 0.0, 0.1743779993845267, 0.0, 0.0, 0.47383102128298327, 0.0]
    """
