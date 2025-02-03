#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 06:10:06 2023

@author: stefanovrizzi
"""

from scipy.stats import pearsonr #for correlation

class CrossCorrelation:
        
    def __init__(self, df_to_join, experiment, export, folder, learning_contexts, method, first_df):
                
        self.df_to_join = df_to_join
        self.experiment = experiment
        self.export = export
        self.folder = folder
        self.learning_contexts = learning_contexts
        self.method = method
        self.first_df = first_df
        
        self.truncated_index = 13
                            
    def join_dataframes(self):
        
        # Accuracy
        
        df_first = self.prepare_index(self.df_to_join[f'df_{self.first_df}'])
        df_first = df_first[self.learning_contexts] if self.first_df in ['accuracy', 'RT'] else df_first
            
        # Parameters
        df_parameters = self.prepare_index(self.df_to_join['df_parameters'])
            
        # Questionnaires
        df_propensity = self.prepare_index(self.df_to_join['df_propensity'])
        df_clinics = self.prepare_index(self.df_to_join['df_clinics'])
        df_clinics = df_clinics[['Alc', 'Nic', 'Anx', 'Dep']]
    
        # Merge three datasets    
        df_joint = df_first.merge(right=df_parameters, on='id', validate='one_to_one') #accuracy (or alternative measures) with parameters
        df_joint = df_joint.merge(right=df_propensity, on='id', validate='one_to_one') # with questionnaires from propensity scales
        self.df_joint = df_joint.merge(right=df_clinics, on='id', validate='one_to_one') # with questionnaires from clinical scales
        
    def prepare_index(self, df):
    
        '''
        Prepare index
        '''
        
        #extract values of interest, either test = 0; retest = 1; or combined (take mean between experiments)
        if isinstance(self.experiment, int):
            df = df.loc[self.experiment].copy()
            
        if self.experiment == 'combined':
        
            df = df.groupby('id').mean()        
        
        if self.experiment == 'difference':
            df = df.unstack('exp').swaplevel(axis=1)[1]-df.unstack('exp').swaplevel(axis=1)[0]
        
        idx = df.reset_index()['id'].astype(str).apply(lambda x: x[:self.truncated_index]).tolist()
        index_dict = dict(zip(df.index.get_level_values('id').tolist(), idx))
        df = df.rename(index=index_dict)
    
        return df
    
    def pearsonr_pvalue(self, x, y):
        '''
        Function to compute p-values
        '''
        return pearsonr(x,y)[1]
    
    def df_cross_correlation(self):
                
        self.join_dataframes()
        
        # Correlation matrices
        R = self.df_joint.corr() #correlation matrix with R values
        
        # P-values
        n = len(self.df_joint.columns) #number of variables in joint dataset, to then apply Bonferroni correction
        
        p_value = self.df_joint.corr(method=self.pearsonr_pvalue)
        significance = p_value.apply(lambda x: x < .05/(n*(n-1)/2)) #apply Bonferroni
        significance_NotCorrected = p_value.apply(lambda x: x < .05) #Not corrected

        filename = f'df_cross_correlation_exp_{self.experiment}_method_{self.method}_R_first_df_{self.first_df}'         
        self.export(R.reset_index(), self.folder, filename)

        filename = f'df_cross_correlation_exp_{self.experiment}_method_{self.method}_pvalue_first_df_{self.first_df}'
        self.export(p_value.reset_index(), self.folder, filename)

        filename = f'df_cross_correlation_exp_{self.experiment}_method_{self.method}_significance_first_df_{self.first_df}'
        self.export(significance.reset_index(), self.folder, filename)
 
        filename = f'df_cross_correlation_exp_{self.experiment}_method_{self.method}_significance_NotCorrected_first_df_{self.first_df}'
        self.export(significance_NotCorrected.reset_index(), self.folder, filename)
            
        return
