"""
Created on Thu Apr 18 15:12:13 2019
@author: Helene Ta

This file regroups every functions we need in our main program. 
Most of those functions are issued from Alberto's and Ilaria's programs. 
I just regroup there and modify a little bit by myself just to use them easier.
Except the discretization function, I create a new one. 
"""
import inspyred
import numpy as np
import pandas as pd
import pyAgrum as gum
import math 
import random
import sys
import os
import csv
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.lib.ipython as gnb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

#################################DISCRETIZATION################################        
def discretization (df, variables_in_order, thresholds):
    discretized = np.zeros((df.shape[0], len(variables_in_order)), dtype = int)
    
    for var_index, var in enumerate(variables_in_order):
        var_matrix = df[var].as_matrix()
        discretization_list = thresholds[var]
        
        for compteur_1 in range (0, var_matrix.shape[0]):
            for compteur_2 in range (0, len(discretization_list)):
                if var_matrix[compteur_1] > discretization_list[compteur_2][0] and var_matrix[compteur_1] <= discretization_list[compteur_2][1]:
                    discretized[compteur_1, var_index] = compteur_2
    
    return discretized

def create_discretized_data (ds, parcelles, variables_in_order):
    final_matrix = np.hstack((parcelles, ds))
    df = pd.DataFrame(data = final_matrix, columns = [["parcelle", "time", "sequence"] + variables_in_order])
    df.to_csv("discretized_data_optimized_hta.csv", sep = ",", index = False)
    
    return

def create_data_agrum(file, chemical_var, var_var, var_in_order_2, var_ts, var_t):
    header = ["parcelle", "sequence", "time"]
    weeks = 3
    df = pd.read_csv(file)
    unique_ts = df[var_ts].unique()
    
    for var in chemical_var: header.append(var + "_0")
    for var in chemical_var: header.append(var + "_t")
    with open("discretized_data_agrum_optimized_hta.csv","w", newline='') as fp: 
        w = csv.writer(fp)
        w.writerow(header)
        
        for ts in unique_ts: 
            selection = df[df[var_ts] == ts].copy()
            time_instants = selection[var_t].unique()
            
            contain_six = False
            contain_six = 6 in time_instants
            
            if len(selection) >= weeks and contain_six:
                for t in range(7-weeks, 7-1):
                    row_0 = selection[selection["time"] == t]
                    row_t = selection[selection["time"] == t + 1]
                    newline = list()
                    newline += list(row_0[var_in_order_2].as_matrix().ravel())
                    newline += list(row_0[var_var].as_matrix().ravel())
                    newline += list(row_t[var_in_order_2].as_matrix().ravel())
                    newline += list(row_t[var_var].as_matrix().ravel())
                    
                    parc = selection["parcelle"].values[0]
                    newline = [parc, ts, t] + newline
                    w.writerow(newline)        

    return

def create_data_agrum_2(file, chemical_var, var_var, var_in_order_2, var_ts, var_t):
    header = ["parcelle", "sequence", "time"]
    weeks = 3
    df = pd.read_csv(file)
    unique_ts = df[var_ts].unique()
    
    for var in chemical_var: header.append(var + "_0")
    for var in chemical_var: header.append(var + "_t")
    with open("discretized_data_agrum_optimized_hta.csv","w", newline='') as fp: 
        w = csv.writer(fp)
        w.writerow(header)
        
        for ts in unique_ts: 
            selection = df[df[var_ts] == ts].copy()
            time_instants = selection[var_t].unique()
            
            contain_six = False
            contain_six = 6 in time_instants
            
            if len(selection) >= weeks and contain_six:
                for t in range(7-weeks, 7-1):
                    row_0 = selection[selection["time"] == t]
                    row_t = selection[selection["time"] == t + 1]
                    newline = list()
                    newline += list(row_0[var_in_order_2].as_matrix().ravel())
                    newline += list(row_0[var_var].as_matrix().ravel())
                    newline += list(row_t[var_in_order_2].as_matrix().ravel())
                    newline += list(row_t[var_var].as_matrix().ravel())
                    
                    parc = selection["parcelle"].values[0]
                    newline = [parc, ts, t] + newline
                    w.writerow(newline)        

    return

#############################DYNAMIC BAYESIAN NETWORKS#########################
def create_dbn (ac_class, ac_m_class, s_class, var_ac_class, var_ac_m_class, var_s_class, ins_class, pl_class, hr_class, t_class):
    dbn = gum.BayesNet()
    
    ac_0 = dbn.add(gum.LabelizedVariable("ac_0", "ac_0", ac_class))
    ac_m_0 = dbn.add(gum.LabelizedVariable("ac_m_0", "ac_m_0", ac_m_class))
    s_0 = dbn.add(gum.LabelizedVariable("s_0", "s_0", s_class))
    var_ac_0 = dbn.add(gum.LabelizedVariable("var_ac_0", "var_ac_0", var_ac_class))
    var_ac_m_0 = dbn.add(gum.LabelizedVariable("var_ac_m_0", "var_ac_m_0", var_ac_m_class))
    var_s_0 = dbn.add(gum.LabelizedVariable("var_s_0", "var_s_0", var_s_class))
    ins_0 = dbn.add(gum.LabelizedVariable("ins_0", "ins_0", ins_class))
    pl_0 = dbn.add(gum.LabelizedVariable("pl_0", "pl_0", pl_class))
    hr_0 = dbn.add(gum.LabelizedVariable("hr_0", "hr_0", hr_class))
    t_0 = dbn.add(gum.LabelizedVariable("t_0", "t_0", t_class))
    
    ac_t = dbn.add(gum.LabelizedVariable("ac_t", "ac_t", ac_class))
    ac_m_t = dbn.add(gum.LabelizedVariable("ac_m_t", "ac_m_t", ac_m_class))
    s_t = dbn.add(gum.LabelizedVariable("s_t", "s_t", s_class))
    var_ac_t = dbn.add(gum.LabelizedVariable("var_ac_t", "var_ac_t", var_ac_class))
    var_ac_m_t = dbn.add(gum.LabelizedVariable("var_ac_m_t", "var_ac_m_t", var_ac_m_class))
    var_s_t = dbn.add(gum.LabelizedVariable("var_s_t", "var_s_t", var_s_class))
    ins_t = dbn.add(gum.LabelizedVariable("ins_t", "ins_t", ins_class))
    pl_t = dbn.add(gum.LabelizedVariable("pl_t", "pl_t", pl_class))
    hr_t = dbn.add(gum.LabelizedVariable("hr_t", "hr_t", hr_class))
    t_t = dbn.add(gum.LabelizedVariable("t_t", "t_t", t_class))
    
    dbn.addArc(ac_0, ac_t)
    dbn.addArc(ac_0, var_ac_0)
    dbn.addArc(ac_0, var_ac_t)
    
    dbn.addArc(ac_m_0, ac_m_t)
    dbn.addArc(ac_m_0, var_ac_m_0)
    dbn.addArc(ac_m_0, var_ac_m_t)
    
    dbn.addArc(s_0, s_t)
    dbn.addArc(s_0, var_s_0)
    dbn.addArc(s_0, var_s_t)	
    	
    dbn.addArc(pl_0, var_ac_0)
    dbn.addArc(pl_0, var_ac_m_0)
    dbn.addArc(pl_0, var_s_0)
	
    dbn.addArc(hr_0, var_ac_0)
    dbn.addArc(hr_0, var_ac_m_0)
    
    dbn.addArc(t_0, var_ac_0)
    dbn.addArc(t_0, var_ac_m_0)
    dbn.addArc(t_0, var_s_0)	
	
    dbn.addArc(ins_0, var_s_0)	
	
    dbn.addArc(	var_ac_0, ac_t)	
    dbn.addArc(var_ac_m_0, ac_m_t)	
    dbn.addArc(var_s_0, s_t)	
    
    dbn.addArc(pl_t, var_ac_t)
    dbn.addArc(pl_t, var_ac_m_t)
    dbn.addArc(pl_t, var_s_t)
    
    dbn.addArc(hr_t, var_ac_t)
    dbn.addArc(hr_t, var_ac_m_t)
    
    dbn.addArc(t_t, var_ac_t)
    dbn.addArc(t_t, var_ac_m_t)
    dbn.addArc(t_t, var_s_t)	
	
    dbn.addArc(ins_t, var_s_t)	
    
    dbn.generateCPTs()
    bn_to_pdf(dbn, "DBN.pdf")
    
    return dbn

def bn_to_pdf (bn, file_name):
    temp_file = "temp.dot"
    with open(temp_file, "w") as fp: fp.write(bn.toDot())
    
    os.system("dot -Tpdf temp.dot > " + file_name)
    os.remove(temp_file)
    
    return 

###########################EVOLUTIONARY ALGORITHM##############################
def my_observer(population, num_generations, num_evaluations, args):
    best = max(population)
    print('{0:6} -- {1} : {2}'.format(num_generations, best.fitness, str(best.candidate)))

def my_evaluator (candidates, args) :
    #Get a few values from args
    variables = args["variables"]
    n_classes = args["n_classes"]
    df = args["df"]
    ref_variable = args["ref_variable"]
    chemical = args["chemical"]
    dbn = args["dbn"]
    predicted_var_0 = args["predicted_var_0"]
    variable_0 = args["var_0"]
    read_var_0 = args["read_var"]
    global_pred_rate = args["global_pred_rate"]
    parcelles = args["parcelles"]
    variables_in_order = args["variables_in_order"]
    chemical_var = args["chemical_var"]
    var_var = args["var_var"]
    variables_in_order_2 = args["variables_in_order_2"]
    discret_thresholds = args["discret_thresholds"]
    
    fitness = []
    
    for candidate in candidates :
#        candidate = sorted(candidate)
        lower_bound_errors = 0
        #Conversion a candidate solution into a discretization
        
        for v_index in range(len(variables)) :
            v = variables[v_index]
            maxV = max(df[v])
            discret_thresholds[v] = []
            lower_bound = -np.inf
            for c_index in range(n_classes-1) :
                i = v_index * (n_classes-1) + c_index
                upper_bound = candidate[i]
                if not np.isinf(lower_bound): upper_bound += lower_bound                
                discret_thresholds[v].append([lower_bound, upper_bound])
                lower_bound = upper_bound
#                print("lb: ", lower_bound)
#                if lower_bound > maxV : lower_bound_errors += 1
                
            discret_thresholds[v].append([lower_bound, np.inf])
            
        discret_thresholds[ref_variable] = args["fixed_bounds"]
        candidate_fitness = 0
        if lower_bound_errors > 0 :
            candidate_fitness += -lower_bound_errors
        else: 
            discretized_dataset = discretization(df, variables_in_order, discret_thresholds)
                
            create_discretized_data(discretized_dataset, parcelles, variables_in_order)
            file = "discretized_data_optimized_hta.csv"
            create_data_agrum_2(file, chemical_var, var_var, variables_in_order_2, "sequence", "time")
            
            file_2 = "discretized_data_agrum_optimized_hta.csv"
            training_file = "discretized_training_optimized_hta.csv"
            valid_file = "discretized_validation_optimized_hta.csv"
            dataframe = pd.read_csv(file_2)
            unique_ts = sorted(dataframe["sequence"].unique())
            
            averages = compute_averages(predicted_var_0, discret_thresholds, df, variables_in_order )
            
            predicted_val(dataframe, dbn, predicted_var_0, unique_ts, "sequence", valid_file, training_file, variable_0, read_var_0, global_pred_rate, averages)
            
            predicted_df = pd.read_csv("Predicted_Values_Optimized.csv")
            ts_pred = predicted_df["sequence"].unique()
            candidate_fitness += fitness_distance_one_variable(df, predicted_df, chemical, ts_pred)
    
        fitness.append(candidate_fitness)
    return fitness

def my_generator (random, args) :
    bounder = args["_ec"].bounder
    individual = []
    
    for i in range(len(bounder.lower_bound)):
        half_distribution = random.uniform(bounder.lower_bound[i], bounder.upper_bound[i])
        individual.append(half_distribution)
    
    return individual

def evolutionary_algorithm (seed, pop_size, my_bounder, num_gen, ref_variable, thresholds, n_classes, df, variable, parcelles, dbn, discret_thresholds):
    var_0 = ["ac_0", "ac_m_0", "s_0", "ins_0", "pl_0", "hr_0", "t_0", "var_ac_0", "var_ac_m_0", "var_s_0"]
    variables_in_order = ["ac", "var_ac", "ac_m", "var_ac_m", "s", "var_s", "ins-7", "pl-7", "hr-7", "t-7"]
    chemical = ["s"]
    read_var = ["ins", "pl", "hr", "t"]
    chemical_var = ["ac", "ac_m", "s", "ins", "pl", "hr", "t", "var_ac", "var_ac_m","var_s"]
    var_var = ["var_ac", "var_ac_m","var_s"]
    predicted_var_0 = ["ac", "ac_m", "s", "var_ac", "var_ac_m", "var_s"]
    variables_in_order_2 = ["ac", "ac_m", "s", "ins-7", "pl-7", "hr-7", "t-7"]
    
    global_pred_rate = dict()
    
    prng = random.Random()    # create pseudo-random number generator
    prng.seed(seed)
    
    ea = inspyred.ec.EvolutionaryComputation(prng) 
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
    ea.replacer = inspyred.ec.replacers.plus_replacement
    ea.terminator = [ inspyred.ec.terminators.generation_termination ]
#    ea.observer = inspyred.ec.observers.stats_observer
    ea.observer = my_observer
    
    variables = [variable]
    final_pop = ea.evolve(  generator = my_generator, 
                            evaluator = my_evaluator, 
                            pop_size = pop_size, 
                            num_selected = 2 * pop_size,
                            bounder = my_bounder,
                            maximize = False,
                            max_generations = num_gen,
                            # here are the values that will go in 'args'
                            fixed_bounds = discret_thresholds[ref_variable],
                            ref_variable = ref_variable,
                            variables = variables,
                            n_classes = n_classes,
#                            num_elites = 2,
                            df = df, 
                            dbn = dbn, 
                            predicted_var_0 = predicted_var_0,
                            var_0 = var_0,
                            read_var = read_var,
                            global_pred_rate = global_pred_rate,
                            parcelles = parcelles, 
                            variables_in_order = variables_in_order, 
                            chemical_var = chemical_var, 
                            var_var = var_var, 
                            variables_in_order_2 = variables_in_order_2, 
                            chemical = chemical,
                            discret_thresholds = discret_thresholds
                            )
    
    return final_pop

#################################METRIC ERROR##################################
def deprobabilize (av, p):
    value = 0
    
    if len(p) != len(av):
        print("Error: they don't have the same size !")
        sys.exit(-1)
    else: 
        for compteur in range (0, len(p)):
            value += av[compteur] * p[compteur]
            
    return value

####################################VALUES#####################################
def compute_averages (predicted_var_0, thresholds, df, variables_in_order) :
    df_reduced = df[variables_in_order]
    average = dict()
    minima = dict()
    maxima = dict()
    
    for v in predicted_var_0 :
        minima[v] = min(df_reduced[v])
        maxima[v] = max(df_reduced[v])
    
    for variable in predicted_var_0 : 
        average[variable]=list()
        for i in range(0,len(thresholds[variable])):
            if thresholds[variable][i][0]!= -np.inf and thresholds[variable][i][1]!= np.inf:
                average[variable].append(0.5*(thresholds[variable][i][0]+thresholds[variable][i][1]))
            elif thresholds[variable][i][0]== -np.inf :
                average[variable].append(0.5*(minima[variable]+thresholds[variable][i][1]))
            elif thresholds[variable][i][1]== np.inf :
                average[variable].append(0.5*(thresholds[variable][i][0]+maxima[variable]))

    return average

def predicted_val (dataframe, dbn, predicted_var_0, unique_time_series, var_ts, valid_file, training_file, variable_0, read_var_0, global_pred_rate, averages):
    with open("Predicted_Values_Optimized.csv", "w", newline = '') as fp:
        header = ["parcelle", "sequence"]
        for variable in predicted_var_0 :
            for i in range (1, 3):
                variable_t = variable + "_" + str(i)
                header.append(variable_t)
        w = csv.writer(fp)
        w.writerow(header)
        
        for index, ts in enumerate(unique_time_series):
            valid_df = dataframe[dataframe[var_ts] == ts].copy()
            valid_df = valid_df.reset_index()
            training_df = dataframe[dataframe[var_ts] != ts].copy()
            training_df = training_df.reset_index()
            
            valid_df.to_csv(valid_file, index = False)
            training_df.to_csv(training_file, index = False)
            
            learner = gum.BNLearner(training_file, dbn)
            learner.setInitialDAG(dbn.dag())
            learner.useScoreLog2Likelihood()
            learner.useAprioriSmoothing(0.01)
        
            dbn_2 = learner.learnParameters(dbn.dag())
            
            steps = 3

            bn = gdyn.unroll2TBN(dbn_2, steps)
#            bn_to_pdf(bn, "DBN_unrolled_2.pdf")
            
            dictionary = dict()
            for variable in variable_0:
                tempdf = valid_df.loc[0,:]
                dictionary[variable] = int(tempdf[variable])
            
            original_values = dict()
            for variable in predicted_var_0: 
                original_values[variable] = []
                
            global_pred_rate[index] = dict()
            pred_row = list(valid_df[["parcelle", "sequence"]].loc[0,:])
            
            for recorder in range (0, steps - 1):
                for variable in read_var_0: 
                    row = valid_df.loc[recorder, :]
                    variable_name = variable + "_" + str(recorder + 1)
                    var_name_in_df = variable + "_t"
                    dictionary[variable_name] = int(row[var_name_in_df])
                for variable in predicted_var_0: 
                    var_name_in_df = variable + "_t"
                    original_values[variable].append(int(row[var_name_in_df]))
                    
            inference = gum.LazyPropagation(bn)
            inference.setEvidence(dictionary)
            inference.makeInference()
            
            pred_values = dict()
            predicted_values = dict()
            
            for variable in predicted_var_0:
                predicted_values[variable] = []
            for variable in predicted_var_0:    
                pred_values[variable] = dict()
                for i in range (1, steps):
                    variable_t = variable + "_" + str(i)
                    predicted_values = []
                    predicted_values = inference.posterior(bn.idFromName(variable_t))[:]
                    index_used = original_values[variable][i - 1]
                    t_list = [0] * len(predicted_values)
                    t_list[index_used] = 1
                    pred_values[variable][variable_t] = deprobabilize(averages[variable], predicted_values)
                    pred_row.append(pred_values[variable][variable_t])
            
            w.writerow(pred_row)
            
#    gnb.showPotential(bn.cpt("s_2"))
    return 

def human_readable(thresholds, variables, n_classes, bestIndividual, upper_bound):
    for v_index in range(len(variables)) :
        v = variables[v_index]
        thresholds[v] = []
        lower_bound = -np.inf
        
        for c_index in range(n_classes - 1) :
            i = v_index * (n_classes - 1) + c_index
            upper_bound = bestIndividual[i]
            
            if not np.isinf(lower_bound) : upper_bound += lower_bound
            
            thresholds[v].append([lower_bound, upper_bound])
            lower_bound = upper_bound
        thresholds[v].append([lower_bound, np.inf])
        
    print("Result of the evolution:")
    for v in thresholds : print("\tVariable \"" + v + "\":", thresholds[v])
        
    return thresholds

def fitness_distance_one_variable(df, predicted_df, chemical_var, ts_pred):
    distance = dict()
    weeks = [5, 6]
#    RMSE = dict()
    fitness = 0
#    header = ["parcelle", "date", "sequence"]
#    all_var= []
#    with open("errors_simple.csv", "w", newline='') as file:        
#    for variable in chemical_var:
#        for t in range (1,3):
#            all_var.append(variable + "_" + str(t))
#            header.append(variable + "_" + str(t)) 
#
#        w = csv.writer(file)
#        w.writerow(header)
    
    for ts in ts_pred:
#        newline = list()
        distance[ts] = dict()
        
        selection_seq = df[df["sequence"] == ts].copy()
        selection_pred = predicted_df[predicted_df["sequence"] == ts].copy()
        
        for week in weeks:
            distance[ts][week - 4] = dict()
            selection = selection_seq[selection_seq["time"] == week].copy()
#            halfrow = list(selection[["parcelle","date","sequence"]].values[0])
            
            for variable in chemical_var:
                distance[ts][week - 4][variable] = abs(selection_pred[variable + "_" + str(week - 4)].values[0] - selection[variable].values[0])
#                    newline.append(distance[ts][week - 4][variable])
                fitness += distance[ts][week - 4][variable]

#            newline = halfrow + newline
#            w.writerow(newline)
#        
#    df_errors = pd.read_csv("errors_simple.csv")
#    for var in all_var:
#        RMSE[var] = 0
#        
#        for index, row in df_errors.iterrows() :
#            RMSE[var] += df_errors[var][index] * df_errors[var][index]
#        RMSE[var]=math.sqrt(RMSE[var]/len(df_errors[var]))
##        if RMSE[var] < 1: fitness += RMSE[var]
#    print("RMSE: ", RMSE)
            
    return (fitness/len(distance))

def create_file_pred_obs(df, ts_pred, weeks, chemical_var, predicted_df):
    all_var_1 = []
    all_var_2 = []
#I create a file comparing predicted vs. original data for each variable
    with open("predicted_vs_observed_hta.csv", "w", newline='') as second_file:
        header = ["parcelle","date","sequence"]
        for variable in all_var_1:
            header.append(variable)
            predicted_variable = "pred" + variable
            all_var_2.append(predicted_variable), header.append(predicted_variable)

        w = csv.writer(second_file)
        w.writerow(header)
        
        for ts in ts_pred:
            newline = list()
            selection_seq_2 = df[df["sequence"] == ts].copy()
            selection_pred_2 = predicted_df[predicted_df["sequence"] == ts].copy()
            
            for week in weeks:
                selection_2 = selection_seq_2[selection_seq_2["time"] == week].copy()
                half_row = list(selection_2[["parcelle","date","sequence"]].values[0])
                
                for variable in chemical_var:
                    newline.append(selection_2[variable].values[0])
                    newline.append(selection_pred_2[variable+"_"+str(week-4)].values[0])
                    
            newline = half_row + newline
            w.writerow(newline)
        
    return

def RMSE_sklearn(all_var_1, observed_values, predicted_values):   
    RMSE = dict()
    total_error = dict()
    new_RMSE = dict()
    R_squared = dict()
    EV = dict()
    df_error = pd.read_csv("absolute_errors_2.csv")
#Here, I compute the RMSE and the average simple error
    for i in range(1, 3):
        for variable in all_var_1: #voir sous quel forme est all_var_1 pour l'introduire ici
            RMSE[variable] = 0
            total_error[variable] = 0
            
            for index, row in df_error.iterrows():
                total_error[variable] += df_error[variable][index]
                RMSE[variable] += df_error[variable][index] * df_error[variable][index]
            
            total_error[variable] = total_error[variable]/len(df_error[variable])
            RMSE[variable] = math.sqrt(RMSE[variable] / len(df_error[variable]))

    #here I compute the R^2 using the sklearn function
    #I also doublecheck the RMSE using sklearn: all good, it's the same of mine 
    for variable in all_var_1:
        R_squared = r2_score(observed_values[variable], predicted_values[variable])
        EV[variable] = explained_variance_score(observed_values[variable], predicted_values[variable])
        new_RMSE[variable] = math.sqrt(mean_squared_error(observed_values[variable], predicted_values[variable]))

    return