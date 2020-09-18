from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [-3, -3, -12, 0, 0]                 #Primalproblem
    b = [1, 1]                        
    A = [[1, 0, 2, 1, 0],
         [0, 1, 1, 0, 1]]
    
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

    z: -7.5
    Indices: [3 2 1 4 5]
    x_B: [0.5 0.5]

    Sensitivity:

    Reduced costs: [1.5 4.5 3. ]
    Delta c: [(-inf, 3.0), (-3.0, 3.0)]
    Delta b: [(-1.0, 1.0), (-0.5, inf)]
    Shadow prices: [-168.03655809874007, -2.745119685818893]

    """


    
