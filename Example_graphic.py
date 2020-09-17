from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-1, -1, 0, 0]              #Example graphic
    b = [4, 6]
    A = [[1, 0.5, 1, 0],    
         [0.5, 1, 0, 1]]

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

    z: -6.666666666666667
    Indices: [1 2 4 3]
    x_B: [1.33333333 5.33333333]

    Sensitivity:

    Reduced costs: [0.66666667 0.66666667]
    Delta c: [(-1.0, 0.5), (-1.0, 0.5)]
    Delta b: [(-0.9999999999999998, 8.000000000000002), (-4.000000000000001, 1.9999999999999996)]
    Shadow prices: [0.0, 0.0]
    """
