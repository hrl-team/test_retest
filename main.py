#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:04:47 2022

@author: svrizzi
"""
import sys 
sys.path.append("/home/svrizzi/v27_36/venvStefano36/lib/python3.6/site-packages")

import os
import argparse

from plot_figures import Plot, Figure2, Figure3, Figure4, Figure5, FigureReactionTime, DemographicsTable, ICCBootstrap, TotalVarianceExplained, Fig4CompareMethods, Figure5CompareNSessions, SummaryStats

def parameter_free_plots(folder):

    for reliability_measure in reliability_measure_list:
        
        fig2 = Figure2(reliability_measure, folder)
        #fig2.plot_sessions()
        #fig2.plot_RT()
        #fig2.plot_main_contrasts()   
        #fig2.plot_main_contrast_reaction_time()   

        #[ fig2.plot_pca(df_name) for df_name in ['df_accuracy', 'df_propensity', 'df_clinics'] ]
        
        #fig3 = Figure3(reliability_measure, folder)
        #fig3.plot()

    #FigureReactionTime(folder).plot()
    DemographicsTable(folder).approved()

def parameter_plots(method):

    for reliability_measure in reliability_measure_list:

        fig2 = Figure2(reliability_measure, method)            
        #fig2.plot(method)
        #fig2.plot_pca('df_parameters', method)
        
        #Realibility measure only controls subplot A in in fig4.plot(), not in other plotting functions. Subplots are generally cross-correlation, not cross-ICC 
        #fig4 = Figure4(reliability_measure, method) 
        #fig4.plot()
        #fig4.plot_difference()
        
        #fig4 = Figure4(reliability_measure, method, first_df='RT')
        #fig4.plot()
        #fig4.plot_difference()
        
        #fig4 = Figure4(reliability_measure, method, first_df='scores_joint')
        #fig4.plot_difference()
        #It does not make sense to plot CV vs reliability for PCA scores, as mean is 0 in PCs
                
        if method in ['ML', 'MAP']:
            fig5 = Figure5(reliability_measure, method)    
            fig5.plot()
        
    ICCBootstrap(folder=method).plot(method=method)

def describe_stats(method):

    sumstats = SummaryStats(method)
    sumstats.describe('df_accuracy')
    sumstats.describe('df_RT')
    sumstats.describe('df_parameters', method)
    sumstats.describe('df_clinics')
    sumstats.describe('df_propensity')
    
def describe_stats_reliability(method, reliability_measure):
    
    sumstats = SummaryStats(method)    
    sumstats.reliability_simulations(reliability_measure, method)

def describe_stats_alpha_con_disc(method):
    
    sumstats = SummaryStats(method)    
    sumstats.ttest_alpha_con_disc(method)

def extreme_upperbound():
    
    method = 'MAP'
    #[ Figure5(reliability_measure, method=method, extreme=True).plot() for reliability_measure in reliability_measure_list]
    [ Figure5(reliability_measure, method=method, extreme=True, split=True).plot_split() for reliability_measure in reliability_measure_list]
    [ Figure5(reliability_measure, method=method, extreme=False, split=True).plot_split() for reliability_measure in reliability_measure_list]   
    #[ Figure_S5(reliability_measure, method=method, extreme=True).plot() for reliability_measure in reliability_measure_list]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Read SLURM_ARRAY_TASK_ID from environment variables
    slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)  # Default to '0' if not set

    # Add SLURM_ARRAY_TASK_ID as an argument
    parser.add_argument('--task_id', type=int, default=int(slurm_array_task_id), 
                        help='Task ID from SLURM_ARRAY_TASK_ID environment variable')
    
    args = parser.parse_args()
    print(args.task_id)
                
    reliability_measure_list = ['ICC', 'correlation']
    
    folder = 'parameter_free'    
    #parameter_free_plots(folder)
    
    method_list = ['ML', 'MAP', 'HB', 'HBpool']
    method_index = list(range(0, len(method_list))) #task_id starts from 0
    method_dict = dict( zip(method_index, method_list) )
    
    method = method_dict[ args.task_id ]
    print(f'\n {args.task_id} {method} \n')

    #parameter_plots(method)
    
    #describe_stats(method)
    #[describe_stats_reliability(method, reliability_measure) for reliability_measure in reliability_measure_list]
    #describe_stats_alpha_con_disc(method)
    
    #[Figure5CompareNSessions(reliability_measure, method).plot() for reliability_measure in reliability_measure_list]

    #extreme_upperbound()
    
    ##method_list = ['HB', 'HBpool']
    folder = 'Joint'
    #[TotalVarianceExplained(folder).plot(exp, method_list) for exp in [0, 1, 'combined' ]]
    #ICCBootstrap(folder=folder).plot_combined(method_list)
    
    method_list = ['HB', 'HBpool']
    [Figure2(reliability_measure, folder).plot_rebuttal_fig14(folder, method_list) for reliability_measure in reliability_measure_list]
    
    #[ Fig4CompareMethods(folder, reliability_measure).reliability_vs_CV() for reliability_measure in reliability_measure_list]
    
    
