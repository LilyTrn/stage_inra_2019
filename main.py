"""
Created on Thu Apr 18 15:01:18 2019

@author: Helene Ta

Compilation of Optimization and DBN program. 
"""
import inspyred
import numpy as np
import pandas as pd
import sys
import functions_main
import time

###############################################################################
def main():
    needed_data = dict()
    needed_data["parcelle"] = list()
    needed_data["s"] = list()
    needed_data["var_s"] = list()
    
    ac_class = 4
    ac_m_class = 5
    s_class = 8
    var_ac_class = 4
    var_ac_m_class = 5
    var_s_class = 4
    ins_class = 5
    pl_class = 6
    hr_class = 5
    t_class = 5
    thresholds = dict()
    thresholds_redefine = dict()
    lower_bounds = []
    upper_bounds = []
    
    seed = 10
    pop_size = 100
    num_gen = 50

    thresholds["var_s"] =[[0,12], [12,20],[20,35],[35, np.inf]]
    thresholds["ac"]=[[-np.inf, 5.47], [5.47, 6.33], [6.33, 7.94], [7.94, np.inf]]
    thresholds["ac_m"]=[[-np.inf, 3.66], [3.66, 4.6], [4.6, 5.7], [5.7, 6.88], [6.88, np.inf]]

    thresholds["var_ac"]= [[-np.inf,-1.5],[-1.5, -1],[-1,-0.6],[-0.6,np.inf]]
    thresholds["var_ac_m"]= [[-np.inf, -2.5], [-2.5, -1.5], [-1.5, -0.75],[-0.75,-0.5],[-0.5,np.inf]]
    thresholds["ins-7"]=[[15,30],[30,40,],[40,55],[55,60],[60,75]]
    thresholds["pl-7"]=[[0,10],[10,20],[20,30],[30,45],[45,70],[70,100]]
    thresholds["t-7"]=	[[0,11],[11,15],[15,17],[17,19.5],[19.5,22]]
    thresholds["hr-7"]=[[60,70],[70,75],[75,80],[80,90],[90,100]]

    dbn = functions_main.create_dbn(ac_class, ac_m_class, s_class, var_ac_class, var_ac_m_class, var_s_class, ins_class, pl_class, hr_class, t_class)
    
    df = pd.read_csv("2019_NewIlariaDataPC.csv", sep=";")
#    df = df.dropna()
    parcelles = df[["parcelle","time","sequence"]].as_matrix()

    minima_s = min(df["s"])
    maxima_s = max(df["s"])
    
    # now, the first minimum value for generation is the lowest value of that variable;
    lower_bounds.append(minima_s)
    lower_bounds.extend([0.5 for i in range(s_class - 2)]) # but the others MUST be positive
    upper_bounds.extend([maxima_s for i in range(s_class - 1)])
    
    lower_bound = lower_bounds 
    upper_bound = upper_bounds
    my_bounder = inspyred.ec.Bounder(lower_bound, upper_bound)
    
    final_pop = functions_main.evolutionary_algorithm(seed, pop_size, my_bounder, num_gen, "var_s", thresholds, s_class, df, "s", parcelles,dbn, thresholds)

    bestIndividual = final_pop[0].candidate
    
    thresholds_redefine = functions_main.human_readable(thresholds, ["s"], s_class, bestIndividual, upper_bound)

    return
    
###############################################################################
if __name__ == "__main__":
    print("It's working.")
    time_start = time.time()
    sys.exit(main())
    time_end = time.time()
    print("It's the end.")
    print("Execution time: %" %((time_end - time_start)/3600))