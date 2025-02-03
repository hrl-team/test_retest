#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:36:15 2020

@author: stefano
"""

#%% Libraries

import os
import shutil
from scipy.io import loadmat
import pandas as pd
import json
import numpy as np

from demographics import Demographics
from setup_data import Setup
from regression import Regression
from fit import FitRL
from HB_fit import HierarchicalBayesian
from reliability_analysis import ReliabilityAnalysis
from cross_correlation import CrossCorrelation
from simulations import Simulations
from setup_PCA import PCA_Setup

import matplotlib.pyplot as plt

class Save:
    
    def __init__(self, folder):
        
        self.folder = folder
        self.path = os.path.join(os.getcwd(), self.folder)
        self.make_folder(self.path)
                
        self.folder_demographics = self.folder_name('Demographics')
        self.folder_demographics_approved = self.folder_name('Demographics_approved')        
        self.folder_setup = self.folder_name('Setup')
        self.folder_regression = self.folder_name('Regression')
        self.folder_reliability = self.folder_name('Reliability')
        self.folder_parameters = self.folder_name('RL_parameters')

        self.folder_simulations = self.folder_name('Simulations')      
        self.folder_cross_correlation = self.folder_name('Cross_correlation')   
        self.folder_pca = self.folder_name('PCA')
        
        self.folder_stats_summary = self.folder_name('Stats_summary')
        self.folder_ttests = self.folder_name('ttests')

        self.folder_figures = self.folder_name('Figures')  
        self.folder_HB = os.path.join(self.folder_figures, 'Hierarchical_Bayesian')

    def folder_name(self, folder_name):
                        
        return os.path.join(self.path, f'{folder_name}/')

    #% Create folders to save figures folder
    
    def make_folder(self, folder_name):
    
        '''
        Create specified folder if it does not exist
        '''
        
        os.mkdir(folder_name) if not os.path.isdir(folder_name) else None
    
    
    def export(self, df, folder, file_name, index=False):
    
        '''
        Export csv data files
        '''    
    
        self.make_folder(folder) #check if folder exists; if not, create it
    
        return df.to_csv(folder+file_name+'.csv', index=index)
    
    ###############################
    #% Save figure in figure folder
    
    def figure(self, title, bbox_inches='tight', HB=False):
        
        '''
        Save paper figures
        '''
        
        figure_format = 'svg'
        #figure_format = 'png'
        
        folder_figures = self.folder_figures
        self.make_folder(folder_figures) #check if folder exists; if not, create it

        folder_figures = self.folder_figures if not HB else self.folder_HB
        self.make_folder(folder_figures)

        plt.savefig(folder_figures+title+'.'+figure_format, format=figure_format, bbox_inches=bbox_inches, dpi=300)
        plt.close()
        
        return
        
#%% #####################################################
############################## IMPORT ###################
        
class Fetch:

    def __init__(self, Save):
                
        self.datafile = "raw_data.mat" #data file name
        self.data_folder = os.getcwd()+'/Data/' 
        self.save = Save
        self.demographics = Demographics(self.save)
        self.setup = Setup(self.save)
        self.regression = Regression(self.save)
        
        self.plain_to_greek_dict = {'alpha3': r'$\alpha_{v}$',
                   'alpha_v': r'$\alpha_{v}$',
                   'alpha_conf': r'$\alpha_{CON}$',
                   'alpha_disc': r'$\alpha_{DISC}$',
                   'beta': r'$\beta$',
                   'beta_par': r'$\beta$'}

        self.greek_to_plain_dict = {r'$\alpha_{v}$': 'alpha3',
                            r'$\alpha_{CON}$': 'alpha_conf',
                            r'$\alpha_{DISC}$': 'alpha_disc',
                            r'$\beta$': 'beta'}
        
        self.df_dict = {'df_accuracy': self.df_accuracy,
                             'df_regression': self.df_regression,
                             'df_propensity': self.df_propensity,
                             'df_clinics': self.df_clinics,
                             'df_parameters': self.df_parameters,
                             'df_simulations': self.df_simulations,
                             'df_accuracy_sessions': self.df_accuracy_sessions,
                             'df_accuracy_simulations': self.df_accuracy_simulations,
                             'df_parameters_simulations': self.df_parameters_simulations,
                             'df_RT': self.df_RT,
                             'df_scores_joint': self.df_scores_joint,
                             'df_accuracy_main_contrasts': self.df_accuracy_main_contrasts,
                             'df_reaction_time_main_contrasts': self.df_reaction_time_main_contrasts
                             }

        self.propensity_dict = {'bis': 'BIS',
                                'bas_drive': 'BASd',
                                'bas_fun': 'BASf',
                                'bas_rewres': 'BASr'}

        self.clinics_dict = {'alcohol': 'Alc',
                             'nicotine': 'Nic',
                             'anxiety': 'Anx',
                             'depression': 'Dep'}
    

    #%% Sequence of outcomes True/Real or Synthetic/Random
    
    def folder_seq_(real_seq):
        
        '''
        Set folder with sequence of True/Real or Synthetic/Random outcomes for RL task
        '''
        
        folder_seq = 'real_seq/' if real_seq == True else 'random_seq/'
    
        return folder_seq
    
    #% Load data
    
    def data(self):
        
        '''load raw data'''
        
        return loadmat(self.data_folder+self.datafile)
    
    #############################
    #% Data structured in a dataframe

    def df_demographics_approved(self):
                
        df_test_L = pd.read_csv(self.data_folder+'Card Game INFL1.csv')
        df_test_R = pd.read_csv(self.data_folder+'Card Game INFR1.csv')
        df_test = pd.concat([df_test_L, df_test_R])
        df_test = df_test.rename(columns={'Age': 'age'})
        
        filepath = self.data_folder+'prolific.csv'
        df_retest = pd.read_csv(filepath)
        df_retest = df_retest.rename(columns={'participant_id': 'Participant id'})
        df = pd.merge(df_test, df_retest, how='inner', on=['Participant id', 'status'])	        
        df = df[df['status']=='Approved']
        df.rename(columns={'Ethnicity (Simplified UK/US Census)': 'Ethnicity (Simplified UK US Census)'}, inplace=True) #avoid conflicts with /
        
        variables = ['sex', 'ethnicity', 'Ethnicity 2', 'Ethnicity (Simplified UK US Census)']
        df[variables] = df[variables].fillna('Unknown')
        
        self.save.export(df, self.save.folder_demographics_approved, 'participants_approved')
        
        Nsubjects = len(df)
        print(f'Demographics approved: N subjects {Nsubjects}')
        
        for variable in variables:
            #df_temp = df.groupby([variable]).count()['Participant id'].apply(lambda x: x/Nsubjects*100).round(1)
            #counts_name = '%'
            
            df_temp = df.groupby([variable]).count()['Participant id'].astype(int)
            counts_name = 'Counts'
            
            self.save.export(df_temp.reset_index().rename(columns={'Participant id': counts_name}), self.save.folder_demographics_approved, f'{variable}_approved')
        
        df['age range'] = df['age_x'].apply(lambda x: '18-30' if x <= 30 else '30-60' if x > 30 and x <= 60 else '60+' if x > 60 else None)
        #df_temp = df.groupby(['sex', 'age range', 'Ethnicity (Simplified UK US Census)']).count()['Participant id'].apply(lambda x: x/Nsubjects*100).round(1)
        #counts_name = '%'
        
        df_temp = df.groupby(['sex', 'age range', 'Ethnicity (Simplified UK US Census)']).count()['Participant id'].astype(int)
        counts_name = 'Counts'
        
        self.save.export(df_temp.reset_index().rename(columns={'Participant id': counts_name}), self.save.folder_demographics_approved, f'detailed_approved')
        
        df_temp = pd.DataFrame()
        df_temp['mean'] = pd.Series( df['age_x'].mean() )
        df_temp['SEM'] = pd.Series( df['age_x'].sem() )
        self.save.export(df_temp.reset_index().rename(columns={'Participant id': '%'}), self.save.folder_demographics_approved, 'age_test_approved')

        df_temp = pd.DataFrame()
        df_temp['mean'] = pd.Series( df['age_y'].mean() )
        df_temp['SEM'] = pd.Series( df['age_y'].sem() )
        self.save.export(df_temp.reset_index().rename(columns={'Participant id': '%'}), self.save.folder_demographics_approved, 'age_retest_approved')
        
        df[['completed_date_time', 'Completed Date Time']] = df[['completed_date_time', 'Completed Date Time']].apply(pd.to_datetime)
        df_temp = (df['completed_date_time'] - df['Completed Date Time']) / np.timedelta64(1, 'D')
        self.save.export(df_temp.describe().reset_index(), self.save.folder_demographics_approved, 'day_difference_approved')
        
        return

    def set_df_index(self, df):
        
        df.set_index(['exp', 'id', 'session', 'round', 'trial'], inplace=True)
        df.sort_index(inplace=True)  
        df.reset_index(inplace=True)
        df.set_index(['exp', 'id'], inplace=True) 
        
        return df

    def df(self):
        
        '''load data structured in a dataframe'''
        
        filepath = self.save.folder_setup+'df.csv'
        
        data = self.data()
        self.setup.df(data) if not os.path.isfile(filepath) else None
        
        df = pd.read_csv(filepath)
        df = self.set_df_index(df)
        
        return df 
    
    def df_accuracy(self):
        
        filepath = self.save.folder_setup+'df_accuracy.csv'

        if not os.path.isfile(filepath):
            
            df = self.df()
            self.setup.df_accuracy(df)
        
        df_accuracy = pd.read_csv(filepath)
        
        df_accuracy['accuracy'] *= 100 #from accuracy 0-1 to percentage      
        df_accuracy = df_accuracy.pivot_table(values='accuracy', columns='learning context', index=['exp', 'id'])
        
        return df_accuracy

    def df_accuracy_main_contrasts(self):
        
        df_contrast_list = []
        
        for variable in ['valence', 'information']:
        
            filename = f'df_accuracy_main_constrasts_{variable}'
            filepath = self.save.folder_setup+filename+'.csv'

            if not os.path.isfile(filepath):
            
                df = self.df()
                self.setup.df_accuracy_main_contrasts(df, variable, filename)
            
            df_contrast = pd.read_csv(filepath)
            df_contrast['variable'] = variable
            df_contrast_list.append( df_contrast )
        
        df_contrast_joint = pd.concat(df_contrast_list)
        df_contrast_joint = df_contrast_joint.set_index(['variable', 'exp', 'id']) #.unstack(['variable', 'sign'])['contrast']
        
        return df_contrast_joint

    def df_reaction_time_main_contrasts(self):
        
        df_contrast_list = []
        
        for variable in ['valence', 'information']:
        
            filename = f'df_reaction_time_main_constrasts_{variable}'
            filepath = self.save.folder_setup+filename+'.csv'

            if not os.path.isfile(filepath):
            
                df = self.df()
                self.setup.df_reaction_time_main_contrasts(df, variable, filename)
            
            df_contrast = pd.read_csv(filepath)
            df_contrast['variable'] = variable
            df_contrast_list.append( df_contrast )
        
        df_contrast_joint = pd.concat(df_contrast_list)
        df_contrast_joint = df_contrast_joint.set_index(['variable', 'exp', 'id']) #.unstack(['variable', 'sign'])['contrast']
        
        return df_contrast_joint
        
    def df_accuracy_simulations(self, method, pool_exp=True, model='ASYMMETRIC RELATIVE', real_seq=True, shuffled=False, extreme=False):
        
        filename = 'df_accuracy_simulations'

        suffix = f'_pool_exp_{pool_exp}_method_{method}_model_{model}_real_seq_{real_seq}'
        suffix += '_shuffled' if shuffled else ''
        suffix += '_extreme' if extreme else ''

        filepath = self.save.folder_simulations+filename+suffix+'.csv'

        if not os.path.isfile(filepath):
        
            #Fetch simulations
            df_simulations = self.df_simulations(method=method, model=model, real_seq=real_seq, shuffled=shuffled, extreme=extreme)
            master_datafile = self.master_datafile()
            simulations = Simulations(self.save.export, self.save.folder_simulations, master_datafile, pool_exp, model, method, real_seq, shuffled, suffix) 
            simulations.df_accuracy(df_simulations)

        df_accuracy = pd.read_csv(filepath)
        
        df_accuracy['accuracy'] *= 100 #from accuracy 0-1 to percentage      
        df_accuracy = df_accuracy.pivot_table(values='accuracy', columns='learning context', index=['exp', 'id'])
    
        return df_accuracy

    def df_accuracy_sessions(self):
        
        filepath = self.save.folder_setup+'df_accuracy_sessions.csv'

        if not os.path.isfile(filepath):
            
            df = self.df()
            self.setup.df_accuracy(df, sessions=True)
        
        df_accuracy = pd.read_csv(filepath)
        df_accuracy['accuracy'] *= 100 #from accuracy 0-1 to percentage
        df_accuracy = df_accuracy.pivot_table(values='accuracy', columns='learning context', index=['exp', 'session', 'id'])
        
        return df_accuracy

    #############################
    #% Data structured in a dataframe
    
    def df_scale(self):
        
        '''load data structured in a dataframe'''
        
        filepath = self.save.folder_setup+'df_scale.csv'
        
        data = self.data()
        self.setup.df_scale(data) if not os.path.isfile(filepath) else None
        
        df = pd.read_csv(filepath)
        df.set_index(['exp', 'id'], inplace=True)
        df.columns = df.columns.rename("variables")
        #df = df.unstack('exp') #experiment as main column level
        #df = df.reorder_levels(['exp', 'variables'], axis=1).reindex([0, 1], level=0, axis=1)
        
        return df
    
    def df_propensity(self):
        
        filepath = self.save.folder_setup+'df_propensity.csv'
        
        columns = ['bas_drive', 'bas_fun', 'bas_rewres', 'bis'] #variables of interest
        
        self.setup.df_scale_of_interest(self.df_scale(), columns, 'df_propensity') if not os.path.isfile(filepath) else None # Create dataframe for questionnaire responses
        
        df = pd.read_csv(filepath)

        df = df.pivot_table(values=columns, index=['exp', 'id'])
                
        columns = self.propensity_dict
        
        df = df.rename(columns=columns)
        
        return df
        
    def df_clinics(self):
        
        filepath = self.save.folder_setup+'df_clinics.csv'
        
        columns = ['alcohol', 'nicotine', 'anxiety', 'depression'] #%% Clinical
        self.setup.df_scale_of_interest(self.df_scale(), columns, 'df_clinics') if not os.path.isfile(filepath) else None
        
        df = pd.read_csv(filepath)

        df = df.pivot_table(values=columns, index=['exp', 'id'])
        
        columns = self.clinics_dict
        
        df = df.rename(columns=columns)
        
        return df
 
    def df_regression(self):
        
        self.save.make_folder(self.save.folder_regression)
        
        filepath = self.save.folder_regression+'df_regression.csv'
        
        df = self.df() #fetch behavioural data
        self.setup.df_regression(df) if not os.path.isfile(filepath) else None

        df = pd.read_csv(filepath)
        
        return df

    def df_reliability(self, reliability_measure, df_name, pool_exp=False, method=None, model='ASYMMETRIC RELATIVE', experiment=None, session=None, real_seq=True, shuffled=False, df_name_pca=None, n_sessions=None, extreme=False, split=False):
        
        self.save.make_folder(self.save.folder_reliability)

        if extreme or n_sessions:
            real_seq = False

        #pool_exp = pool_exp if not extreme else True
        
        df_name += '_'+df_name_pca.split('_')[-1] if df_name == 'df_scores' else ''
        
        filename = f'{df_name}_{reliability_measure}'
        
        suffix = f'_method_{method}' if method is not None else ''
        suffix += f'_model_{model}' if df_name.split('_')[-1] in ['parameters', 'simulations'] else ''
        suffix += f'_exp_{experiment}' if experiment is not None else ''
        suffix += f'_session_{session}' if session is not None else ''
        suffix += '_shuffled' if shuffled else ''
        suffix += '_extreme' if extreme else ''
        suffix += f'_{split}' if split else ''
        suffix += f'_N_sessions_{n_sessions}' if n_sessions else ''
        filename += suffix
        
        filepath = self.save.folder_reliability+filename+'.csv'
        
        simulations = df_name.split('_')[-1] == 'simulations' #check if dataframe is about simulations
        parameters = df_name.split('_')[-1] == 'parameters' #check if dataframe is about parameters   
        
        if not os.path.isfile(filepath):

            if simulations:

                df = self.df_dict[df_name](pool_exp=False, method=method, model=model, real_seq=real_seq, shuffled=shuffled, n_sessions=n_sessions, extreme=extreme)
                
            elif parameters and df_name.split('_')[1] != 'scores':
                
                df = self.df_dict[df_name](method, pool_exp=pool_exp, model=model, drop=True)
                
            elif df_name.split('_')[1] == 'scores':
                
                if df_name_pca.split('_')[1] == 'parameters':

                    df_scores_test = self.df_scores(exp=0, df_name=df_name_pca, method=method)
                    df_scores_retest = self.df_scores(exp=1, df_name=df_name_pca, method=method)

                else:
                    df_scores_test = self.df_scores(exp=0, df_name=df_name_pca)
                    df_scores_retest = self.df_scores(exp=1, df_name=df_name_pca)
                
                df_scores_test['exp'] = 0
                df_scores_retest['exp'] = 1

                df_scores = pd.concat([df_scores_test, df_scores_retest])

                df_scores.reset_index(inplace=True)
                df = df_scores.reset_index().set_index(['exp', 'id']).drop('index', axis=1)

            else:
                df = self.df_dict[df_name]()
                
            if split: #split by median beta fit from pooled data
                
                df_parameters = self.df_dict['df_parameters'](pool_exp=True,method=method)
                condition = df_parameters[r'$\beta$']>df_parameters[r'$\beta$'].median() if split == 'top' else df_parameters[r'$\beta$']<=df_parameters[r'$\beta$'].median() #if split == 'bottom'
                idx = df_parameters[condition].index
                df = df[df.index.get_level_values('id').isin(idx)] #filter by id subselection
            
            reliability_analysis = ReliabilityAnalysis(self.save.export, self.save.folder_reliability, self.master_datafile(), method=method, model=model, n_sessions=n_sessions, extreme=extreme, split=split)
            reliability_analysis.df_reliability[reliability_measure](df, df_name)

        df_reliability = pd.read_csv(filepath)
        
        if simulations:

            reliability_measure_dict = dict({'ICC': 'ICC', 'correlation': 'R'})
            reliability_measure_metric = reliability_measure_dict[reliability_measure]
            df_reliability = df_reliability.pivot_table(values=reliability_measure_metric, columns=df_reliability.columns[0], index=['pair_number'])
            
        else:
            
            df_reliability.set_index(df_reliability.columns[0], inplace=True)
                
            #df_reliability['p*'].fillna('', inplace=True) #to plot significance stars
            df_reliability['p*'] = df_reliability['p*'].fillna('') #to plot significance stars            
            df_reliability['p*'] = df_reliability['p*'].astype(str)            
        
        return df_reliability

    def df_parameters(self, method, pool_exp=False, model='ASYMMETRIC RELATIVE', drop=True, extreme=False):
          
        self.save.make_folder(self.save.folder_parameters)
        
        filename = 'df_parameters'
        
        suffix = f'_pool_exp_{pool_exp}_method_{method}_model_{model}'
        suffix += '_extreme' if extreme else ''
        
        filename += suffix
        
        filepath = self.save.folder_parameters+filename+'.csv'
        
        if not os.path.isfile(filepath):

            master_datafile = self.master_datafile()
            
            if not extreme:

                df = self.df() 
            
                if method in ['ML', 'MAP']: #maximum likelihood or Maximum A Posteriori
                    
                    fit = FitRL(pool_exp, model, master_datafile, self.save.export, self.save.folder_parameters, filename)
                    MAP = (method == 'MAP')
                    fit.ML(df, MAP=MAP)
                        
                elif method in ['HB', 'HBpool']: #Hierarchical Bayesian modelling
            
                    hbModelType = 'independent_normal' #'reduced' #'independent_normal' #'independent'      
                    hb = HierarchicalBayesian(df, method, pool_exp, hbModelType, self.save.folder_parameters, self.save.export)
                    hb.fit()
                    self.df_parameters_HB(method, hb.experiments, filename) # Join test and retest mean parameter values

            elif extreme: #to test test-retest reliability upper limit with argmax and p=1, we start from synthetic parameter generation
                
                fit = FitRL(pool_exp, model, master_datafile, self.save.export, self.save.folder_parameters, filename)
                fit.df_parameters_synthetic()
        
        df_parameters = pd.read_csv(filepath)
        df_parameters = self.prepare_df_parameters(df_parameters, model=model, drop=drop)
        
        return df_parameters
        
    def df_parameters_simulations(self, pool_exp=False, method=None, model='ASYMMETRIC RELATIVE', real_seq=True, shuffled=False, drop=True, n_sessions=None, extreme=False):
                    
        self.save.make_folder(self.save.folder_parameters)

        if extreme or n_sessions:
            real_seq = False
        #pool_exp = pool_exp if not extreme else True

        filename = 'df_parameters_simulations'
        
        suffix = f'_pool_exp_{pool_exp}_method_{method}_model_{model}_real_seq_{real_seq}'
        suffix += '_shuffled' if shuffled else ''
        suffix += '_extreme' if extreme else ''
        suffix += f'_N_sessions_{n_sessions}' if n_sessions else ''
        
        filename += suffix
        
        filepath = self.save.folder_parameters+filename+'.csv'
        
        if not os.path.isfile(filepath):
        
            df = self.df_simulations(method=method, model=model, real_seq=real_seq, shuffled=shuffled, n_sessions=n_sessions, extreme=extreme)

            if method in ['ML', 'MAP']: #maximum likelihood or Maximum A Posteriori
          
                master_datafile = self.master_datafile()
                fit = FitRL(pool_exp, model, master_datafile, self.save.export, self.save.folder_parameters, filename)
                MAP = (method == 'MAP')
                fit.ML(df, MAP=MAP)
                        
            elif method in ['HB', 'HBpool']: #Hierarchical Bayesian modelling
            
                hbModelType = 'independent_normal' #'reduced' #'independent_normal' #'independent'           
                hb = HierarchicalBayesian(df, method, pool_exp, hbModelType, self.save.folder_parameters, self.save.export)
                hb.fit()
                self.df_parameters_HB(pool_exp, hb.experiments, filename) # Join test and retest mean parameter values

        df_parameters = pd.read_csv(filepath)

        df_parameters = self.prepare_df_parameters(df_parameters, model=model, drop=drop) #, extreme=extreme)
        
        return df_parameters

    def df_parameters_shuffled(self, pool_exp, method, model, drop=True, extreme=False):

        self.save.make_folder(self.save.folder_parameters)
        
        filename = 'df_parameters'
        
        suffix = f'_pool_exp_{pool_exp}_method_{method}_model_{model}_shuffled'
        suffix += '_extreme' if extreme else ''
        
        filepath = self.save.folder_parameters+filename+suffix+'.csv'
        
        if not os.path.isfile(filepath):
            
            df = self.df()
            
            df_parameters = self.df_parameters(pool_exp=pool_exp, method=method, model=model, drop=False, extreme=extreme)
            columns = {r'$\alpha_{v}$': 'alpha3',
                           r'$\alpha_{CON}$': 'alpha_conf',
                           r'$\alpha_{DISC}$': 'alpha_disc',
                           r'$\beta$': 'beta'}
            df_parameters = df_parameters.rename(columns=columns)  
            
            master_datafile = self.master_datafile()
            fit = FitRL(df, pool_exp, model, master_datafile, self.save.export, self.save.folder_parameters)           
            fit.df_parameters_shuffled(df_parameters, pool_exp=pool_exp, method=method, model=model)
            
        df_parameters_shuffled = pd.read_csv(filepath)
        
        df_parameters_shuffled = self.prepare_df_parameters(df_parameters_shuffled, model=model, drop=drop)
        
        return df_parameters_shuffled

    def prepare_df_parameters(self, df_parameters, model, drop=True):

        df_parameters.set_index(['exp', 'id'], inplace=True) if 'exp' in df_parameters.columns else df_parameters.set_index(['id'], inplace=True)
                
        columns = self.plain_to_greek_dict
            
        df_parameters = df_parameters.rename(columns=columns)
        df_parameters.columns.rename('parameter', inplace=True)
        
        if drop:           

            df_parameters.drop(['alpha1', 'alpha2'], axis=1, inplace=True) if model == 'ASYMMETRIC RELATIVE' else None
            
        return df_parameters

    def df_simulations(self, method, model, real_seq, shuffled, pool_exp=True, n_sessions=None, extreme=False):
        
        '''
        inputs:
        - RL model to simulate with
        - former parameter fitting method (ML or HB) for parameters to be employed now,
        - shuffle parameters vertically or not,
        - employ outcomes from real experiments
        '''
        
        self.save.make_folder(self.save.folder_simulations)

        real_seq = real_seq if not extreme else False
        #pool_exp = pool_exp if not extreme else True
        
        filename = 'df_simulations'
        
        suffix = f'_pool_exp_{pool_exp}_method_{method}_model_{model}_real_seq_{real_seq}'
        suffix += '_shuffled' if shuffled else ''
        suffix += '_extreme' if extreme else ''
        suffix += f'_N_sessions_{n_sessions}' if n_sessions else ''
        
        filepath = self.save.folder_simulations+filename+suffix+'.csv'
        
        if not os.path.isfile(filepath):
            
            df = self.df()
            master_datafile = self.master_datafile()

            df_parameters = self.df_parameters(pool_exp=pool_exp, method=method, model=model, drop=False, extreme=extreme) if not shuffled else self.df_parameters_shuffled(pool_exp, method, model, drop=True, extreme=extreme)
 
            columns = self.greek_to_plain_dict
            
            df_parameters = df_parameters.rename(columns=columns)     
 
            simulations = Simulations(self.save.export, self.save.folder_simulations, master_datafile, pool_exp, model, method, real_seq, shuffled, suffix, n_sessions=n_sessions, extreme=extreme)            
            simulations.df_simulations(df_parameters, df)
     
        df_simulations = pd.read_csv(filepath)
        df_simulations = self.set_df_index(df_simulations)
        
        return df_simulations
    
    def df_slope(self, df_name, method=None, pool_exp=True, model='ASYMMETRIC RELATIVE', real_seq=True, shuffled=False, extreme=False):

        real_seq = real_seq if not extreme else False
        #pool_exp = pool_exp if not extreme else True
        
        filename = f'{df_name}_slope'
        filename += f'_pool_exp_{pool_exp}_model_{model}_method_{method}_real_seq_{real_seq}' if df_name.split('_')[-1] in ['parameters', 'simulations'] is not None else ''
        filename += '_shuffled' if shuffled else ''
        filename += '_extreme' if extreme else ''
        
        filepath = self.save.folder_regression+filename+'.csv'
        
        if not os.path.isfile(filepath):
            
            simulations = df_name.split('_')[-1] == 'simulations' #check if dataframe is about simulations
            parameters = df_name.split('_')[-1] == 'parameters' #check if dataframe is about parameters
                        
            if simulations:
                
                df = self.df_dict[df_name](pool_exp=pool_exp, method=method, model=model, real_seq=real_seq, shuffled=shuffled, extreme=extreme)
                
            elif parameters:
                
                df = self.df_dict[df_name](method, pool_exp=pool_exp, model=model, drop=True, extreme=extreme)
            
            else:
            
                df = self.df_dict[df_name]() 

            self.regression.linear_fit(df, df_name, simulations+parameters, pool_exp=pool_exp, method=method, model=model, filename=filename, real_seq=real_seq, shuffled=shuffled)
        
        df_regression = pd.read_csv(filepath) 
        df_regression.set_index(df_regression.columns[0], inplace=True)
        
        return df_regression
    
    def df_reliability_vs_CV(self, reliability_measure, df_name, method=None):
        
        filepath = self.save.folder_reliability+f'{df_name}_{reliability_measure}_vs_CV'
        filepath += f'_method_{method}' if method is not None else ''
        filepath += '.csv'
        
        if not os.path.isfile(filepath):
                
            df = self.df_dict[df_name](method=method) if df_name == 'df_parameters' else self.df_dict[df_name]()
            df_reliability = self.df_reliability(reliability_measure, df_name, method=method)     
            reliability_analysis = ReliabilityAnalysis(self.save.export, self.save.folder_reliability, self.master_datafile(), method=method)
            reliability_analysis.df_reliability_vs_CV(df, df_name, df_reliability, reliability_measure)
            
        df_reliability_vs_CV = pd.read_csv(filepath)
        
        df_reliability_vs_CV.set_index(df_reliability_vs_CV.columns[0], inplace=True)
            
        return df_reliability_vs_CV
 
    def df_cross_correlation_dict(self, experiment, method=None, p=True, first_df='accuracy'):

        #first_df = first_df + '_joint' if first_df == 'scores' else first_df
        
        filename = f'df_cross_correlation_exp_{experiment}_method_{method}_R_first_df_{first_df}'
        filepath = self.save.folder_cross_correlation+filename+'.csv'
        
        if not os.path.isfile(filepath):

            learning_contexts = self.master_datafile()['learning_contexts']
            
            df_to_join = {f'df_{first_df}': self.df_dict[f'df_{first_df}'](),
                            'df_parameters': self.df_dict['df_parameters'](method=method),
                            'df_propensity': self.df_dict['df_propensity'](),
                            'df_clinics': self.df_dict['df_clinics']()}

            cross_correlation = CrossCorrelation(df_to_join, experiment, self.save.export, self.save.folder_cross_correlation, learning_contexts, method, first_df)
            cross_correlation.df_cross_correlation()
                    
        R = pd.read_csv(filepath)
        R.set_index(R.columns[0], inplace=True)
        
        if p:
            filepath = self.save.folder_cross_correlation+f'df_cross_correlation_exp_{experiment}_method_{method}_significance_first_df_{first_df}.csv'
            pvalue = pd.read_csv(filepath)
            pvalue.set_index(pvalue.columns[0], inplace=True)
        
            filepath = self.save.folder_cross_correlation+f'df_cross_correlation_exp_{experiment}_method_{method}_significance_NotCorrected_first_df_{first_df}.csv'
            pvalue_NotCorrected = pd.read_csv(filepath)
            pvalue_NotCorrected.set_index(pvalue_NotCorrected.columns[0], inplace=True)
        
        cross_correlation_dict = {'R': R, 'p': pvalue, 'p_NotCorrected': pvalue_NotCorrected} if p else {'R': R}
 
        return cross_correlation_dict
    
    def df_scores(self, exp, df_name, method=None):

        filepath = self.save.folder_pca+f'df_scores_exp_{exp}_{df_name}.csv'

        df = self.df_pca(exp, df_name, filepath, method)

        df.set_index('id', inplace=True)

        return df
        
    def df_scores_joint(self, df_name='df_accuracy'):
    
        filename = f'df_scores_joint_{df_name}'
        folder = self.save.folder_pca
        filepath = folder+filename+'.csv'

        #df_scores_test = self.df_scores(exp=0, df_name=df_name, method=method)
        #df_scores_retest = self.df_scores(exp=1, df_name=df_name, method=method)

        if not os.path.isfile(filepath):
                
            df_scores_test = self.df_scores(exp=0, df_name=df_name)
            df_scores_retest = self.df_scores(exp=1, df_name=df_name)
            
            df_correction = np.sign(df_scores_test.corrwith(df_scores_retest)) #take correlation sign to align scores sign between experiments
            df_scores_retest *= df_correction
                        
            df_scores_test['exp'] = 0
            df_scores_retest['exp'] = 1
            
            df_scores = pd.concat([df_scores_test, df_scores_retest])

            df_scores.reset_index(inplace=True)
            df = df_scores.reset_index().set_index(['exp', 'id']).drop('index', axis=1)
            
            self.save.export(df.reset_index(), folder, filename)

        df = pd.read_csv(filepath)
        df = df.set_index(['exp', 'id'])
            
        return df

    def df_loadings(self, exp, df_name, method=None):

        filepath = self.save.folder_pca+f'df_loadings_exp_{exp}_{df_name}.csv'

        df = self.df_pca(exp, df_name, filepath, method)
        
        df = df.set_index('Principal Component')
        df.columns.name = 'variable' #df_name.split('_')[-1]
        df = df.stack().to_frame().rename(columns={0: 'Loadings'})

        return df

    def df_variance_explained(self, exp, df_name, method=None):

        filepath = self.save.folder_pca+f'df_variance_explained_exp_{exp}_{df_name}.csv'

        df = self.df_pca(exp, df_name, filepath, method)
        
        return df

    def df_pca(self, exp, df_name, filepath, method):
    
        if not os.path.isfile(filepath):
                
            if df_name == 'df_parameters':
                df = self.df_dict[df_name](method=method, pool_exp=False, model='ASYMMETRIC RELATIVE', drop=True)
            else:
                df = self.df_dict[df_name]()

            df = df.xs(exp, level='exp')

            pc_analysis = PCA_Setup(df, exp, df_name, self.save.export, self.save.folder_pca)
            pc_analysis.run()

        df = pd.read_csv(filepath)
        
        return df

    def df_RT(self):

        filepath = self.save.folder_setup+f'df_RT.csv'

        if not os.path.isfile(filepath):
            
            df = self.df()
            df = df.groupby(['exp', 'learning context', 'id']).median()['reaction time'].rename('Reaction time').to_frame()
            self.save.export(df.reset_index(), self.save.folder_setup, 'df_RT')
                        
        df_RT = pd.read_csv(filepath)
        df_RT = df_RT.pivot_table(values='Reaction time', columns='learning context', index=['exp', 'id'])
        
        return df_RT
    
    def df_summary(self, experiment, method):

        filepath = self.save.folder_parameters+f'df_summary_{method}_{experiment}.csv'

        df = pd.read_csv(filepath)

        return df

    def df_fit_HB(self, experiment, method):

        filepath = self.save.folder_parameters+f'df_fit_{method}_{experiment}.csv'

        df = pd.read_csv(filepath)

        return df

    def df_icc_bootstrap(self, method):

        filename = f'df_icc_bootstrap_method_{method}'
        filepath = self.save.folder_reliability+filename+'.csv'
        
        if not os.path.isfile(filepath):

            df_parameters = self.df_parameters(method=method, pool_exp=False, model='ASYMMETRIC RELATIVE', drop=True)

            RA = ReliabilityAnalysis(self.save.export, self.save.folder_reliability, self.master_datafile(), method=method, model='ASYMMETRIC RELATIVE')
            RA.icc_bootstrap(df_parameters, filename)

        df = pd.read_csv(filepath)
        
        df.set_index('index', inplace=True)
        columns = self.plain_to_greek_dict
        df = df.rename(columns=columns)
        
        df.columns.name = 'parameter'
        df = df.stack('parameter').to_frame()
        df.rename(columns={0: 'ICC'}, inplace=True)

        return df
        
    def df_icc_bootstrap_mean(self, method):

        filename = f'df_icc_bootstrap_method_{method}_mean'
        filepath = self.save.folder_reliability+filename+'.csv'

        df = pd.read_csv(filepath)        
        df.set_index('index', inplace=True)
        #df = df.T
        
        index = self.plain_to_greek_dict
        df = df.rename(index=index)
        df.reset_index(inplace=True)
        
        #df.columns.name = 'parameter'
        df.rename(columns={'0': 'ICC', 'index':'parameter'}, inplace=True)

        return df

    def df_fit_HB_mean(self, parameter, experiment, method):

        filepath = self.save.folder_parameters+f'df_fit_{method}_mean_{parameter}_{experiment}.csv'

        df = pd.read_csv(filepath)

        return df
        
    def df_summary_HB(self, experiment, method):
        
        filepath = self.save.folder_parameters+f'df_summary_{method}_{experiment}.csv'
        df = pd.read_csv(filepath)

        return df

    def df_parameters_HB(self, method, experiments, filename):

        df_list = [self.prepare_df_HB(experiment, method) for experiment in experiments]
        df = pd.concat(df_list) if method == 'HB' else df_list[0] if method == 'HBpool' else breakpoint()     
        self.save.export(df, self.save.folder_parameters, filename)

    def prepare_df_HB(self, experiment, method):

        df = self.df_summary_HB(experiment, method)
        subjects = self.master_datafile()['subjects']

        par_example = df['index'].str.startswith('beta_par[')
        experiments = df[par_example]['index'].str.split('[', expand=True)[1].str.split(',', expand=True)[1].str.split(']', expand=True)[0].astype(int)
        experiments += 1*(experiment=='retest') #it is saved as 0 in case of unpooled HB fitting
        Nsubjects = len( df[par_example]['index'].str.split('[', expand=True)[1].str.split(',', expand=True)[0] )        
        index = pd.MultiIndex.from_product([subjects[:Nsubjects], experiments.unique()], names=['id', 'exp'])

        df_par = pd.DataFrame(index = index)

        df_par['beta'] = df[ df['index'].str.split('[', expand=True)[0] == 'beta_par' ]['mean'].values
        df_par['alpha1'] = 0
        df_par['alpha2'] = 0
        df_par['alpha3'] = df[ df['index'].str.split('[', expand=True)[0] == 'alpha_v' ]['mean'].values
        df_par['alpha_conf'] = df[ df['index'].str.split('[', expand=True)[0] == 'alpha_conf' ]['mean'].values
        df_par['alpha_disc'] = df[ df['index'].str.split('[', expand=True)[0] == 'alpha_disc' ]['mean'].values

        df_par.reset_index(inplace=True)
        df_par.rename(columns={'index':'id'}, inplace=True)

        return df_par

    #############################
    #% Master dataframe with key variables (subjects, learning contexts, n trials, etc)
    
    def master_datafile(self):
        
        '''
        Load data from master dataframe
        '''
        
        filepath = self.save.folder_setup+'master_datafile.json'
        
        df = self.df()
        self.setup.master_datafile(df) if not os.path.isfile(filepath) else None
        
        f = open(filepath,) # Opening JSON file
        datafile = json.load(f) # returns JSON object as a dictionary
        f.close() # Closing file
        
        return datafile
    
