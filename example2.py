from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [102, 99, 101, 98, 98, 104, 100, 101, 102, 94]              #Example 2
    b = [12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000]
    A = [[105, 3.5, 5, 3.5, 4, 9, 6, 8, 9, 7],
        [0, 103.5, 105, 3.5, 4, 9, 6, 8, 9, 7],
        [0, 0, 0, 103.5, 4, 9, 6, 8, 9, 7],
        [0, 0, 0, 0, 104, 9, 6, 8, 9, 7],
        [0, 0, 0, 0, 0, 109, 106, 8, 9, 7], 
        [0, 0, 0, 0, 0, 0, 0, 108, 9, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 109, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 107]]
    
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
