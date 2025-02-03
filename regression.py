#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:13:29 2023

@author: stefanovrizzi
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class Regression:
    
    def __init__(self, save):
        
        self.save = save
        
    #%% Slopes and intercepts
    
    def linear_fit(self, df, df_name, simulations, pool_exp, method, model, filename, real_seq=True, shuffled=False):
    
        lr = LinearRegression(fit_intercept=True)
        
        variables_name = df.columns.name
        variables = df.columns
        
        df.reset_index(inplace=True)
        df['pair_number'] = (df['exp']/2).astype(int)
        df['exp'] = df['exp'].apply(lambda x: int(np.mod(x, 2))) #binaries all experiments ids
        
        intercept = np.zeros((df.pair_number.max()+1, len(variables)))
        slope = np.zeros((df.pair_number.max()+1, len(variables)))
        
        for pair_number in df.pair_number.unique(): #even-numbered tests 
            
            df_test = df[df.pair_number==pair_number]
            
            df_test.set_index('exp', inplace=True)
            
            for variable_n, variable in enumerate(variables):
                
                x = np.array(df_test.loc[0][variable].values).reshape(-1, 1)
                y = np.array(df_test.loc[1][variable].values).reshape(-1, 1)
                
                lr_model = lr.fit(x, y)
                intercept[pair_number, variable_n] = lr_model.intercept_[0]
                slope[pair_number, variable_n] = lr_model.coef_[0][0]
                              
        df_slope = pd.DataFrame(data=slope, columns=variables, index=df.pair_number.unique())
        df_slope.index.name = 'pair_number'
        df_slope.columns.name = variables_name
        #df_intercept = pd.DataFrame(data=intercept, columns=columns, index=test_pairs)

        #filename = f'{df_name}_slope'
        #filename += f'_pool_exp_{pool_exp}_model_{model}_method_{method}_real_seq_{real_seq}' if simulations else ''
        #filename += '_shuffled' if shuffled else ''
        
        self.save.export(df_slope.reset_index(), self.save.folder_regression, filename)
        
        return df_slope #, df_intercept
