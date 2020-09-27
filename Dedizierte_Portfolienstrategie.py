from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
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
    A = [[100+7.875/2, 100+2*8.125/2, 100+2*2.625/2, 8.125/2, 2*2.625/2, 2.500/2, 2*2.250/2, 2*1.750/2, 2*1.750/2, 2*1.625/2, 7.125/2, 2*0.125/2, 6.250/2, 2*2.875/2, 2*2.875/2,     # manchmal Faktor 2, weil Kuponausschüttung
          2.500/2, 2*2.250/2, 2*2.500/2, 2.375/2, 7.500/2, 2.000/2, 2*2.125/2, 2.000/2, 2*2.250/2, 2*2.875/2, 2.625/2, 2*2.250/2, 2*1.625/2, 2*1.625/2, 2*6.500/2,                   # noch in Jahr 2020 
          6.625/2, 2*0.625/2, 2*0.500/2, 6.375/2, 2*6.125/2, 2.750/2, 2*2.875/2, 5.500/2, 2*5.250/2, 2*3.125/2],
         [0, 0, 0, 100+8.125/2, 100+2.625/2, 2.500/2, 2.250/2, 1.750/2, 1.750/2, 1.625/2, 7.125/2, 0.125/2, 6.250/2, 2.875/2, 2.875/2,                                  # Kupon/2, da halbjährliche Ausschüttung
          2.500/2, 2.250/2, 2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 100+2.500/2, 100+2.250/2, 100+1.750/2, 1.750/2, 1.625/2, 7.125/2, 0.125/2, 6.250/2, 2.875/2, 2.875/2,
          2.500/2, 2.250/2, 2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 100+1.750/2, 100+1.625/2, 7.125/2, 0.125/2, 6.250/2, 2.875/2, 2.875/2,
          2.500/2, 2.250/2, 2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100+7.125/2, 100+0.125/2, 6.250/2, 2.875/2, 2.875/2,
          2.500/2, 2.250/2, 2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100+6.250/2, 100+2.875/2, 100+2.875/2,
          2.500/2, 2.250/2, 2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          100+2.500/2, 100+2.250/2, 100+2.500/2, 2.375/2, 7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 100+2.375/2, 100+7.500/2, 2.000/2, 2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 100+2.000/2, 100+2.125/2, 2.000/2, 2.250/2, 2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 100+2.000/2, 100+2.250/2, 100+2.875/2, 2.625/2, 2.250/2, 1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100+2.625/2, 100+2.250/2, 100+1.625/2, 1.625/2, 6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100+1.625/2, 100+6.500/2,
          6.625/2, 0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          100+6.625/2, 100+0.625/2, 0.500/2, 6.375/2, 6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 100+0.500/2, 100+6.375/2, 100+6.125/2, 2.750/2, 2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 100+2.750/2, 100+2.875/2, 5.500/2, 5.250/2, 3.125/2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 100+5.500/2, 100+5.250/2, 100+3.125/2]]

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
    
    z: 121330957.5341076
    Indices: [ 2  5  7  9 11 14 17 19 22 25
              27 30 32 35 37 39  1  3  4  6
              8 10 12 13  15 16 18 20 21 23
              24 26 28 29 31 33 34 36 38 40]
    x_B: [27343.35363967 42737.79455966 73298.72811325 74123.33880453
          84771.91801907 87791.9175985  89053.92641398 90055.78308613
          71125.19551028 71880.90071258 72914.18866032 73734.47328275
          56130.84366444 56306.25255089 48030.63153526 48721.07186358]

    Sensitivity:
    
    Reduced costs: [2.0823526  1.90189595 0.17736762 0.53703619 0.44315305 0.31767114
                    0.20468138 0.9921062  0.43       0.60277831 0.03416559 1.46821567
                    0.65118118 0.77692914 0.06717069 0.85472638 0.30136162 1.18214468
                    0.29416656 1.49391492 1.71084399 0.99832042 1.53748356 0.42444753]
    Delta c: [(-7.1434923700822, 2.003824604141299), (-6.2591869955383705, 0.1726804318769515), (-6.178099440383342, 0.4442513196014449),
              (-6.109369038697988, 0.3178680835389051), (-23.686713604384202, 0.21184075081146614), (-22.871902092344442, 0.4299999999999926),
              (-22.547777786661193, 0.034123413497421096), (-22.296937242681025, 1.431952511641504), (-22.035268430073923, 0.651584135262798),
              (-45.53545067919151, 0.0673782605859526), (-44.89015470530279, 0.30229578414023683), (-44.390758670262336, 1.210727220861915),
              (-42.99347086708216, 0.28562451692143725), (-42.859534820767266, 1.5358265033101233), (-41.58596465326114, 0.9989359038998488),
              (-inf, 0.42888790504524493)]
    Delta b: [(-2956500.112289542, inf), (-4329872.811325465, 114106825.76241304), (-7412333.880452877, 134622282.14428025),
              (-7477191.801906841, 174600292.00248343), (-8779191.759849772, 91901599.55757509), (-8905392.641397612, 109628634.15579008),
              (-9005578.308613336, 141656944.14838794), (-9112519.55102812, 294584882.6029598), (-7188090.071257792, 152407839.70245317),
              (-7291418.866032122, 114268606.12039094), (-7373447.328274984, 147652496.81125793), (-7613084.366443922, 52771570.25456208),
              (-5630625.255089059, 550539406.6807189), (-5803063.153526162, 57897926.12350163), (-4872107.1863581, 125120879.64194877),
              (-5000000.0, 70317189.59163447)]
Shadow prices: [0.9748904538808909, 0.9931737188484275, 0.9896581074820784, 0.9901260403358951,
                0.9937553733729722, 0.9839066939866331, 0.986275078794821, 0.9913502669646106,
                0.9826305192428568, 0.978290126643025, 0.9734109469406032, 0.9598047192252168,
                0.9695571264688835, 0.9512284603289256, 0.9525951830078313, 0.9421724354197938]

    """
