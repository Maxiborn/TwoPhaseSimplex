from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-5, -10, 10, 0]                #Example 2PS
    b = [7, 3]
    A = [[1, 1, -1, 0],
         [1, -2, 2, 1]]
    
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

    z: -70.0
    Indices: [2 4 3 1]
    x_B: [ 7. 17.]

    Sensitivity:

    Reduced costs: [0. 5.]
    Delta c: [(-inf, 5.0), (-inf, 1.6666666666666667)]
    Delta b: [(-7.0, inf), (-17.0, inf)]
    Shadow prices: [-13.062971218311393, -0.0]
    """
    
