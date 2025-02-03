#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:16:52 2022

@author: svrizzi
"""

import pandas as pd
from scipy.stats import pearsonr #for correlation
import pingouin as pg
import numpy as np

#%% Draw correlation table between variables of interest, adding * for chosen significance level #########################

class ReliabilityAnalysis():

    def __init__(self, export, folder, master, method=None, model=None, n_sessions=None, extreme=False, split=False):    

        self.export = export
        self.folder = folder
        self.learning_contexts = master['learning_contexts']
        
        self.method = method #fitting method from which RL parameters where obtained
        self.model = model

        self.significance_level = [0.05, 0.01, 0.001] #one * for first entry, ** for second, etc, to indicate p-value significance level in correlation table

        self.df_reliability = dict({'correlation': self.df_correlation,
                                       'ICC': self.df_icc})
        
        self.ICC_type = 'ICC2' #Choose ICC2, also known as ICC(A, 1), absolute agreement
        
        self.n_sessions = n_sessions
        self.extreme = extreme
        self.split = split
               
    def pearsonr_pval(self, x,y):
        return pearsonr(x,y)[1]
    
    def correlation_routine(self, df, df_name, experiment=None, session=None):
        
        df_correlation = pd.DataFrame(index=df.columns)
        df_correlation.index.name = df.columns.name       
        df_correlation['R'] = df.loc[0].corrwith(df.loc[1])
        df_correlation['R^2'] = df_correlation['R']**2
        df_correlation['p'] = df.loc[0].corrwith(df.loc[1], method=self.pearsonr_pval)
        df_correlation['p*'] = df_correlation['p'].apply(lambda x: ''.join(['*' for t in self.significance_level if x<=t]))
        df_correlation.index.name = df.columns.name
        
        simulations = (df_name.split('_')[-1] == 'simulations') #check if dataframe comes from simulations
        
        if not simulations:

            filename = self.correlation_routine_filename(df_name, experiment, session)
            
            self.export( df_correlation.reset_index(), self.folder, filename)

        return df_correlation
    
    def df_correlation(self, df, df_name):

        if df_name == 'df_accuracy_sessions':
            
            [self.correlation_routine(df.xs(experiment, level='exp'), df_name, experiment=experiment) for experiment in range(2)]
            [self.correlation_routine(df.xs(session, level='session'), df_name, session=session) for session in range(2)]
            
        elif df_name.split('_')[-1] == 'simulations': 
            
            df.reset_index(inplace=True)
            
            df['pair_number'] = (df['exp']/2).astype(int)
            df['exp'] = df['exp'].apply(lambda x: int(np.mod(x, 2))) #binaries experiment ids
            
            df.set_index(['pair_number', 'exp', 'id'], inplace=True)
            
            df_correlation_list = []
            
            for pair_number in df.index.get_level_values('pair_number').unique():
                
                df_experiment = df.xs(pair_number, level='pair_number')
                
                df_correlation_temp = self.correlation_routine(df_experiment, df_name)
                
                df_correlation_temp['pair_number'] = pair_number
                
                df_correlation_list.append( df_correlation_temp ) #append with dataframe
            
            df_correlation = pd.concat(df_correlation_list)
            
            filename = self.reliability_filename(df_name, 'correlation')
            
            self.export( df_correlation.reset_index(), self.folder, filename)

        elif df_name.split('_')[-1] == 'contrasts': 
        
            variables = df.index.get_level_values('variable').unique()
        
            df_correlation_list = []
        
            for variable in variables:
                
                df_temp = df.xs(variable, level='variable').rename(index={-1:0}) #rename just to deal with loc[0] in _routine
                df_correlation_temp = self.correlation_routine(df_temp, f'{df_name}_{variable}')

                df_correlation_temp.index = df_correlation_temp.index.map({'contrast':variable})
                
                df_correlation_list.append( df_correlation_temp ) #append with dataframe
            
            df_correlation = pd.concat(df_correlation_list)
            
            filename = self.reliability_filename(df_name, 'correlation')
            
            self.export( df_correlation.reset_index(), self.folder, filename)
        
        else: #any other dataframe
        
            self.correlation_routine(df, df_name)
        
        return
    
    def icc_routine(self, data, df, df_name, raters='exp', experiment=None, session=None):

        variables = df.columns
        
        icc_results = dict(zip(variables, []*len(variables)))
        
        for variable in variables:

            df_iccs = pg.intraclass_corr(data=data, targets='id', raters=raters, ratings=variable) #compute ICCs
            
            filename = self.ICC_routine_filename(df_name, variable, experiment, raters, session)
            
            self.export( df_iccs, self.folder, filename) #export all statistics
            
            df_iccs.set_index('Type', inplace=True)
            
            icc_results[variable] = dict({'ICC': df_iccs.loc[self.ICC_type]['ICC'],
                                          'pval': df_iccs.loc[self.ICC_type]['pval'],
                                          'p*': ''.join(['*' for alpha in self.significance_level if df_iccs.loc[self.ICC_type]['pval']<=alpha])}) #add * according to significance level                   

        df_icc = pd.DataFrame().from_dict(icc_results, orient='index')
        df_icc.index.name = df.columns.name
        
        return df_icc

    def df_icc(self, df, df_name):

        data = df.reset_index()
        
        if df_name == 'df_accuracy_sessions':
            
            for index in range(2):
            
                data_experiment = data[data.exp==index]
                df_icc = self.icc_routine(data_experiment, df, df_name, raters='session', experiment=index)
                self.export( df_icc.reset_index(), self.folder, f'{df_name}_ICC_exp_{index}')
        
                data_experiment = data[data.session==index]
                df_icc = self.icc_routine(data_experiment, df, df_name, raters='exp', session=index)
                self.export( df_icc.reset_index(), self.folder, f'{df_name}_ICC_session_{index}')
        
        elif df_name.split('_')[-1] == 'simulations': 

            data['pair_number'] = (data['exp']/2).astype(int)
            
            df_icc_list = []
            
            for pair_number in data.pair_number.unique():
                
                data_experiment = data[data.pair_number == pair_number]
                
                df_icc_temp = self.icc_routine(data_experiment, df, df_name)
                df_icc_temp['pair_number'] = pair_number
                
                df_icc_list.append( df_icc_temp ) #append with dataframe
            
            df_icc = pd.concat(df_icc_list)
            
            filename = self.reliability_filename(df_name, 'ICC')
        
            self.export( df_icc.reset_index(), self.folder, filename)

        elif df_name.split('_')[-1] == 'contrasts': 
        
            variables = df.index.get_level_values('variable').unique()
        
            df_icc_list = []
        
            for variable in variables:
                
                df_temp = df.xs(variable, level='variable')
                df_icc_temp = self.icc_routine(df_temp.reset_index(), df_temp, f'{df_name}_{variable}')

                df_icc_temp.index = df_icc_temp.index.map({'contrast':variable})
                
                df_icc_list.append( df_icc_temp ) #append with dataframe
            
            df_icc = pd.concat(df_icc_list)

            filename = self.reliability_filename(df_name, 'ICC')
        
            self.export( df_icc.reset_index(), self.folder, filename)       

        else: #any other dataframe
        
            df_icc = self.icc_routine(data, df, df_name)
            
            filename = self.reliability_filename(df_name, 'ICC')
        
            self.export( df_icc.reset_index(), self.folder, filename)
        
        return

    def correlation_routine_filename(self, df_name, experiment, session):

        filename = f'{df_name}_correlation'
        filename += f'_method_{self.method}' if self.method is not None else ''
        filename += f'_model_{self.model}' if df_name.split('_')[-1] in ['parameters', 'simulations'] else ''
        filename += f'_exp_{experiment}' if experiment is not None else ''
        filename += f'_session_{session}' if session is not None else ''
        filename += '_extreme' if self.extreme else ''
        filename += f'_{self.split}' if self.split else ''
        filename += f'_N_sessions_{self.n_sessions}' if self.n_sessions else ''
                
        return filename

    def ICC_routine_filename(self, df_name, variable, experiment, raters, session):

        filename = f'{df_name}_{variable}_ICC'
        filename += f'_method_{self.method}' if self.method is not None else ''
        filename += f'_model_{self.model}' if df_name.split('_')[-1] in ['parameters', 'simulations'] else ''
        filename += f'_exp_{experiment}' if raters == 'sessions' else ''
        filename += f'_session_{session}' if session is not None else ''
        filename += '_extreme' if self.extreme else ''
        filename += f'_{self.split}' if self.split else ''        
        filename += f'_N_sessions_{self.n_sessions}' if self.n_sessions else ''
        
        return filename

    def reliability_filename(self, df_name, reliability_measure):
    
        filename = f'{df_name}_{reliability_measure}'
        filename += f'_method_{self.method}' if self.method is not None else ''
        filename += f'_model_{self.model}' if df_name.split('_')[-1] in ['parameters', 'simulations'] is not None else ''
        filename += '_extreme' if self.extreme else ''
        filename += f'_{self.split}' if self.split else ''
        filename += f'_N_sessions_{self.n_sessions}' if self.n_sessions else ''
        
        return filename

    def df_reliability_vs_CV(self, df, df_name, df_reliability, reliability_measure):
                
        reliability_measure_temp = 'R' if reliability_measure == 'correlation' else reliability_measure
        #breakpoint()
        df_reliability_vs_CV = pd.DataFrame({'CV': df.std()/df.mean(), f'{reliability_measure_temp}': df_reliability[f'{reliability_measure_temp}']})
                
        filename = f'{df_name}_{reliability_measure}_vs_CV'
        filename += f'_method_{self.method}' if self.method is not None else ''
        filename += '_extreme' if self.extreme else ''
        filename += f'_{self.split}' if self.split else ''
        
        self.export(df_reliability_vs_CV.reset_index(), self.folder, filename)


    def icc_bootstrap(self, df_parameters, filename, num_experiments=2):
        
        print(f'ICC bootstrap {filename}')

        parameters = df_parameters.columns.tolist()
        Nparameters = len(parameters)
        subjects = df_parameters.index.get_level_values('id').unique()
        num_subjects = len( subjects )
        
        #bootstrap over subjects (with replacement)
        bootstraps = 1000
        subjects_idx_bootstrap = np.random.choice(subjects, size=[Nparameters, bootstraps, num_subjects])
      
        data = np.zeros((Nparameters, bootstraps, num_subjects, num_experiments))
        
        #data to compute ICC on
        for parameter_n, parameter in enumerate(parameters):
            for bootstrap in range(bootstraps):
                for idn, idx in enumerate(subjects_idx_bootstrap[parameter_n, bootstrap]):
                    for exp in range(num_experiments):
                        
                        data[parameter_n, bootstrap, idn, exp] = df_parameters.loc[exp, idx][parameter] #.unique() #stan starts at 1

        icc = np.zeros((bootstraps, Nparameters))
        index = pd.MultiIndex.from_product([subjects, range(num_experiments)], names=['id', 'exp'])

        for parameter_n, parameter in enumerate(parameters):

            for bootstrap in range(bootstraps):
            
                df = pd.DataFrame(data=np.reshape( data[parameter_n, bootstrap], -1), index=index, columns=[parameter]).reset_index()

                df_iccs = pg.intraclass_corr(data=df, targets='id', raters='exp', ratings=parameter) #compute ICCs
                df_iccs.set_index('Type', inplace=True)
                icc[bootstrap, parameter_n] = df_iccs.loc[self.ICC_type]['ICC']

        df_icc = pd.DataFrame(data=icc, index=range(bootstraps), columns=parameters)
        self.export(df_icc.reset_index(), self.folder, filename)
            
        self.export(df_icc.mean(axis=0).reset_index(), self.folder, filename+'_mean')
        
        return
