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



    """
    Output:

    Results:
    z: 93944.50353724477
    Indices: [ 1  3  4  5  6  8  9 10  2  7]
    x_B: [ 62.13612744 125.24293381 151.5050805  156.80775832 123.08006865
           124.15727483 104.08985681  93.45794393]

    Sensitivity:
    Reduced costs: [0.83061224 8.78684   ]
    Delta c: [(-61.04999999999927, inf), (-360.828597142857, 0.8426501035196587),
              (-343.6462829931972, inf), (-332.0253941963259, inf),
              (-inf, 9.035524152657636), (-inf, inf), (-inf, inf), (-inf, inf)]
    Delta b: [(-6524.293380736334, inf), (-13150.50804977315, 137010.16099546303),
              (-15680.77583151521, 202579.3094718632), (-16308.006864775822, 184347.17161939552),
              (-13415.727482605642, 89305.9631400627), (-13408.985681214095, 108506.74521517618),
              (-11345.794392523365, 105130.97980848182), (-10000.0, 144630.19079366856)]
    Shadow prices: [0.971502843860996, 0.9156357847896147, 0.8830439065646478, 0.8357613778370265,
                    0.6563908561921767, 0.6194548558644405, 0.5326975429767588, 0.5242884194236035]
    """
