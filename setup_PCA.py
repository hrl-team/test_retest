#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:20:49 2022

@author: svrizzi
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler #standardise before PCA

#%% PCA function to extract all useful variables at once ##############################################################################

class PCA_Setup:
    
    def __init__(self, df, exp, df_name, export, folder):
        
        self.df = df
        self.exp = exp
        self.df_name = df_name
        self.export = export
        self.folder = folder

        self.n_components = 4 #choose number of components
        
        self.pca_dict_keys = ['loadings', 'scores', 'variance_explained', 'variance_explained_ratio']
        
        self.standardise = True #whether to standardise data or not before PCA
        self.standardise = False if df_name == 'df_accuracy' else self.standardise
        print(f'PCA - standardise? {self.standardise}')
        
    def run(self):
        
        #'''
        #PCA on accuracy. Running PCA on the matrix of the average per subject and condition
        #'''
                       
        pca_dict = dict(zip(self.pca_dict_keys, self.pc(self.df)))
        self.pc_export(pca_dict) 
    
    def pc(self, df):
        
        PC_list = ['PC'+str(i) for i in range (1, self.n_components+1)]
        loadings, score, variance_explained, variance_explained_ratio = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  
    
        pca = PCA(n_components=self.n_components, svd_solver='full')
        
        if 'exp' in df.columns.names: #if experiments have not been pooled together
        
            tests = df.columns.get_level_values('exp').unique()
        
            for test in tests:
                
                X = StandardScaler().fit_transform(df[test]) if self.standardise else df[test].copy()
                pca.fit(X)
        
                #loadings dataframe        
                loadings_temp = pd.DataFrame(data=pca.components_, columns=df[test].columns)
                loadings_temp['Principal Component'] = PC_list
                loadings_temp['exp'] = test
                loadings = pd.concat([loadings, loadings_temp])
                
                #score dataframe
                score_temp = pd.DataFrame(pca.transform(X), columns=PC_list)
                score_temp['id'] = df.index.get_level_values('id')
                score_temp['exp'] = test
                score = pd.concat([score, score_temp])
                
                # variance explained dataframe
                variance_explained_temp = pd.DataFrame(pca.explained_variance_, columns=['Explained variance']) #also explained_variance_ratio_
                variance_explained_temp['exp'] = int(test)
                variance_explained_temp['Principal Component'] = PC_list
                variance_explained = pd.concat([variance_explained, variance_explained_temp], levels='exp')
                
                # variance explained ratio dataframe
                variance_explained_ratio_temp = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained variance ratio'])
                variance_explained_ratio_temp['exp'] = int(test)
                variance_explained_ratio_temp['Principal Component'] = PC_list
                var_ratio = pd.concat([variance_explained_ratio, variance_explained_ratio_temp], levels='exp')
            
            #Set indeces to dataframes
            index = ['exp', 'Principal Component']
            loadings = loadings.set_index(index)
            score = score.set_index( ['exp', 'id'] )
            variance_explained = variance_explained.set_index(index)
            variance_explained_ratio = variance_explained_ratio.set_index(index)
            
        else: #if experiments have been pooled together
                
            X = StandardScaler().fit_transform(df.copy()) if self.standardise else df.copy()
            pca.fit(X)
        
            #loadings dataframe        
            loadings = pd.DataFrame(data=pca.components_, columns=df.columns)
            loadings['Principal Component'] = PC_list
                
            #score dataframe
            score = pd.DataFrame(pca.transform(X), columns=PC_list)
            score['id'] = df.index.get_level_values('id')
                
            # variance explained dataframe
            variance_explained = pd.DataFrame(pca.explained_variance_, columns=['Explained variance']) #also explained_variance_ratio_
            variance_explained['Principal Component'] = PC_list
                
            # variance explained ratio dataframe
            variance_explained_ratio = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained variance ratio'])
            variance_explained_ratio['Principal Component'] = PC_list
            
            #Set indeces to dataframes
            index = 'Principal Component'
            loadings = loadings.set_index(index)
            score = score.set_index('id')
            variance_explained = variance_explained.set_index(index)
            variance_explained_ratio = variance_explained_ratio.set_index(index)
                            
        return loadings, score, variance_explained, variance_explained_ratio

    def pc_export(self, pca_dict):
        
        for (pca_value_name, pca_variable) in zip(pca_dict.keys(), pca_dict.values()):
        
            file_name = f'df_{pca_value_name}_exp_{self.exp}_{self.df_name}'
            self.export(pca_variable.reset_index(), self.folder, file_name)
