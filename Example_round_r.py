from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-2.000000000000001, -1, 0, 0]           #rounding example (right solution)
    b = [1, 3]                                  
    A = [[0, 1, 1, 0],
         [2, 1, 0, 1]]
    
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

    z: -3.0000000000000013
    Indices: [3 1 2 4]
    x_B: [1.  1.5]    #right basisvalues

    Sensitivity:

    Reduced costs: [4.4408921e-16 1.0000000e+00]
    Delta c: [(-inf, 4.440892098500626e-16), (-inf, 8.881784197001252e-16)]
    Delta b: [(-1.0, inf), (-3.0, inf)]
    Shadow prices: [0.0, -0.8366073576084085]
    """


    
