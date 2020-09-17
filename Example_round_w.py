from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-2.0000000000000001, -1, 0, 0]  #rounding example (wrong solution)
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
    z: -3.0
    Indices: [2 1 3 4]
    x_B: [1. 1.]           # wrong basisvalues

    Sensitivity:

    Reduced costs: [0. 1.]
    Delta c: [(-inf, inf), (-inf, 2.0)]
    Delta b: [(-1.0, 2.0), (-2.0, inf)]
    Shadow prices: [0.0, 0.0]
    """


    
