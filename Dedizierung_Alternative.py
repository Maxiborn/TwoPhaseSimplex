from two_phase_simplex import TwoPhaseSimplex
import numpy as np

if __name__ == "__main__":

    #Add Input in Standardform here:
    
    c = [103.41, 105.41, 101.95,    #Nicht zeitgebundene Dedizierung
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
    b = [121988455.6613,  520445033.8252,  2758097721.6363]
    A = [[103.8752, 108.0208, 102.5245, 107.9874, 103.7628, 103.5755, 104.2934, 103.2493, 104.0896, 103.7228, 
          117.4010, 99.9413, 118.2236, 109.5618, 109.3690, 108.0602, 108.1536, 108.9138, 108.2229, 128.6312, 
          107.2256, 108.5232, 107.6448, 109.8734, 113.2829, 111.9191, 110.8958, 107.0454, 107.7594, 138.9648, 
          139.6382, 101.4210, 100.3211, 140.9186, 141.9238, 116.7324, 118.7729, 139.1780, 139.6288, 121.8734],
         [51.9363, 82.8826, 79.9365, 105.9521, 131.4745, 131.3273, 153.0550, 181.7757, 203.7867, 233.0371, 
          275.6799, 277.9408, 331.2218, 313.5622, 344.2723, 341.9374, 362.7466, 394.8186, 416.2216, 462.0531, 
          464.4612, 495.4489, 515.6726, 549.8315, 559.2891, 555.5060, 572.7753, 592.5102, 614.8201, 747.2244, 
          778.9769, 644.8807, 717.6555, 842.0972, 874.0902, 803.5637, 838.8782, 949.4748, 979.6355, 904.1613],
         [25.9688, 64.5765, 62.6537, 104.9424, 168.2108, 168.0639, 228.4757, 323.0761, 405.4293, 529.6203, 671.5010,
          774.0360, 966.5233, 928.2739, 1114.6420, 1109.1675, 1254.1220, 1472.6629, 1640.3560, 1770.0655,
          2060.6616, 2333.3001, 2537.9648, 2852.3008, 2887.0476, 2873.1489, 3089.6538, 3377.3428, 3632.3952, 4432.3277,
          4766.4107, 4164.9188, 5199.0083, 5538.1397, 5981.3083, 5838.8789, 6311.1997, 7137.8335, 7630.7310, 7203.4178]]
         

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

    z: 120516129.38047802
    Indices: [14 30  2 39  3  4  5  6  7  8
               9 10 11 12 13  1 15 16 17 18
              19 21 22 23 24 25 26 27 28 29
              20 31 32 33 34 35 36 37 38 40]
    x_B: [306073.98264989 556669.87836918 102729.48242756]

    Sensitivity:

    Reduced costs: [0.62424123 1.89218094 1.92391089 1.51748461 1.72025626 1.0605057 1.34341281 0.86501475 1.06716746
                    1.18730362 1.15588462 1.06190056 2.28512786 0.5035091  0.91909402 0.40122319 0.59125256 1.22086345
                    1.42142849 1.07777325 1.66915853 1.11515377 1.07368521 1.42827325 0.68181464 1.12344988 0.7735748
                    2.84648272 1.42716737 0.87644954 1.32816572 1.37710125 0.10358245 1.37513809 0.85571055 1.78078333
                    0.93574519]
    Delta c: [(-0.10270685420319943, 0.39298545367964394), (-13.687336303478661, 0.06657797450856764),
              (-3.19497801552953, 0.30894252161272373)]
    Delta b: [(-7100301.811810486, 46152007.58556704), (-31347911.494123567, 13861109.100537512),
              (-129085748.65134421, 212832029.16388196)]
    Shadow prices: [0.9687903535505692, 0.009983750296163146, -0.0010373464822760242]
    """