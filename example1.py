from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-1.0, -1.0, 0.0, 0.0, 0.0]           #Example 1
    b = [5.0, 1.0, 4.0]                       
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
