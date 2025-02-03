#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 03:38:59 2023

@author: stefanovrizzi
"""

import pandas as pd
import numpy as np

from decimal import Decimal #to shrink subject ids

class Setup:
    
    def __init__(self, save):
        
        self.save = save

        self.learning_contexts_dict = {"RP": 1,"PP": 2, "RC": 3, "PC": 4}        
        self.exp_dict = {0: "test", 1: "retest"}

    # Create dataframe for behavioural task
	    
    def df(self, data):
            
        ColNames = [ #Variables' names
    		'id',
    		'expid',
    		'session', # the 2x2 design was repeated twice'
    		'trial',
    		'round', #state which condition are we (1,2,3,4)
    		'p1', #reward probability of option 1
    		'p2', #reward probability of option 2
    		'magnitude', #magnitude (not manipulated here)
    		'valence', #(reward or punishment)
    		'inf_original', #(partial or complete)
    		'choice_original', #(which option has been choosen)
    		'outcome', # outcome of the chosen option
    		'outcome_unchosen', # outcome of the forgone option - applicable only in the complete case)
    		'reaction time'
    		]
    	    
        learning_data_1 = pd.DataFrame(data=data['learning_data_1RT'], columns=ColNames)
        learning_data_1['exp']= 0 #'test'
        learning_data_2 = pd.DataFrame(data=data['learning_data_2RT'], columns=ColNames)
        learning_data_2['exp']= 1 #'retest'
    	    
        df = pd.concat([learning_data_1, learning_data_2], ignore_index=True)
        
        df['id'] = df['id'].apply(lambda x: Decimal(x) )
        
        #!!! Modify recorded magnitude
        df.magnitude = .5
        df.outcome = df.outcome*df.magnitude #change from -1/1 to -.5/.5
        df.outcome_unchosen = df.outcome_unchosen*df.magnitude #change from -1/1 to -.5/.5
    	    
        #Add useful columns
        df['information']=(df['inf_original']-.5)*2 #the inforation factor :-1 part, 1 compt
        df['information'] = df['information'].astype(int)
        df['valence'] = df['valence'].astype(int)
            
        df['interaction']=df['valence']*df['information'] #the interaction
            
        df['choice']=df['choice_original'].astype(int)-1 #from 1,2 to 0,1 for suboptimal/optimal option
    		
        # As in Nature Communications paper
        conditions = [
                (df['valence']==1) & (df['information']==-1), # reward partial (green condition in the paper) 
                (df['valence']==-1) & (df['information']==-1), #punishment partial (red condition in the paper)
                (df['valence']==1) & (df['information']==1), # reward complete (blue condition in the paper)
                (df['valence']==-1) & (df['information']==1)] # punishment complete (purple condition in the paper)

        keys = self.learning_contexts_dict.keys() #RP, PP, RC, PC #check self.learning_contexts_dict
            
        df['learning context'] = np.select(conditions, keys)
        df['learning context number'] = df['learning context'].map(self.learning_contexts_dict) #learning contexts
        df['round'] = df['round'].astype(int)
        df['trial'] = df['trial'].apply(lambda x : int(np.mod(x, 20))) # better for plotting
        df['session'] = df['session'].astype(int)
        df['reaction time']= df['reaction time']/1000 #may be useful to look at
        df['per'] = (df['choice']-.5)*2 # -1/1 for suboptimal/optimal as dummy variable for GLM
    	    
        #df.set_index(['exp', 'session', 'learning context', 'id'], inplace=True) #structure dataframe by experiment, session and subject id
        #df.sort_index(inplace=True)

        # Save in folder setup
        self.save.export(df, self.save.folder_setup, 'df')
	    
        return df

    def df_accuracy(self, df, sessions=False):
        
        '''
        Compute accuracy dataframe
        '''
        
        groupby_columns = ['exp', 'id', 'learning context'] if not sessions else ['exp', 'session', 'id', 'learning context']

        df_accuracy = df.reset_index().groupby(groupby_columns)['choice'].mean().rename('accuracy') # compute accuracy from choices
        
        filename = 'df_accuracy'
        filename += '_sessions' if sessions else ''
        
        self.save.export(df_accuracy.reset_index(), self.save.folder_setup, filename)
        
        return df_accuracy

    def df_accuracy_main_contrasts(self, df, variable, filename):
        
        '''
        Compute accuracy contrasts dataframe
        '''
        
        groupby_columns = ['exp', 'id', f'{variable}']

        df_accuracy = df.reset_index().groupby(groupby_columns)['choice'].mean().to_frame().rename(columns={'choice':'accuracy'}) # compute accuracy from choices
        df_accuracy['accuracy'] *= 100 #from accuracy 0-1 to percentage  
        df_accuracy = df_accuracy.unstack(variable)['accuracy']
                
        df_contrasts = (df_accuracy[1]-df_accuracy[-1]).to_frame().rename(columns={0:'contrast'})
        
        self.save.export(df_contrasts.reset_index(), self.save.folder_setup, filename)
        
        return

    def df_reaction_time_main_contrasts(self, df, variable, filename):
        
        '''
        Compute response time contrasts dataframe
        '''
        
        groupby_columns = ['exp', 'id', f'{variable}']
                
        df_RT = df.reset_index().groupby(groupby_columns)['reaction time'].median().to_frame().rename(columns={'reaction time':'Reaction time (s)'}) # compute accuracy from choices 
                
        df_RT = df_RT.unstack(variable)['Reaction time (s)']
                
        df_contrasts = (df_RT[1]-df_RT[-1]).to_frame().rename(columns={0:'contrast'})
        
        self.save.export(df_contrasts.reset_index(), self.save.folder_setup, filename)
        
        return

    def accuracy_diff(self, df_accuracy):
        
        #Accuracy difference (retest - test)
        df = df_accuracy[1]-df_accuracy[0] # input accuracy dataframe and test numbers to be subtracted (0 for test, 1 for retest)
        df = self.accuracy_diff[self.learning_contexts_dict.keys()] #re-order columns
        
        return df 

	#%% Create dataframe for questionnaire responses

    def df_scale(self, data):
            
        ColNames = [ #Variables' names
    
    		        'id',
    		        'expid',
    		        'brexit_leave',
    		        'nicotine',	
    		        'alcohol',	
    		        'anxiety',
    		        'depression',
    		        'bis',
    		        'bas_drive',
    		        'bas_fun',
    		        'bas_rewres',	
    		        'childhood_pun',	
    		        'childhood_chaos',	
    		        'childhood_ses',
    		        'adult_ses',
    		        'cepremap',
    		        'trust',
    		        'time_taken',	
    		        'age',
    		        'ses_prolific',	
    		        'sex_num',
    		        'education_num',	
    		        'employment_num',
    		        'income_num',
    		        'smoker_num',
    		        'tobacco_ecigarettes_num',	
    		        'smoking_frequency_num',
    		        'units_alcohol_num',
    		        'alcohol_therapy_num',	
    		        'bmi_num',
    		        'weekly_excercise_num',	
    		        'medication_num',
    		        'long_term_health_condition_num',
    		        'chronic_disease_num',
    		        'cognitive_impairment_num',	
    		        'mental_health_support_num',
    		        'mental_illness_daily_impact_num',	
    		        'mental_illness_ongoing_num',
    		        'political_affiliation_num']
            
        scale_data_1 = pd.DataFrame(data=data['scales_data_1'], columns=ColNames)
        scale_data_1['exp']= 0 #'test'
        scale_data_2 = pd.DataFrame(data=data['scales_data_2'], columns=ColNames)
        scale_data_2['exp']= 1 #'retest'
        
        df = pd.concat([scale_data_1, scale_data_2], ignore_index=True)
        df = df.drop(['expid'], axis=1)

        df['id'] = df['id'].apply(lambda x: Decimal(x) )
         
    	### Save in folder setup
        self.save.export(df.reset_index(), self.save.folder_setup, 'df_scale')

    def df_scale_of_interest(self, df_scale, columns_of_interest, df_name):
        
        df = df_scale.pivot_table(values=columns_of_interest, index=['exp', 'id']) #.astype(int) #restrict dataset to variables of interest          
        
        ### Save in folder setup
        self.save.export(df.reset_index(), self.save.folder_setup, df_name)
 
    def df_regression(self, df):
        
        from sklearn import linear_model #General Linear Model
        from tqdm import tqdm #track loop progress
        import pandas as pd
        
        regressors=['baseline', 'valence','information','interaction']
        model = linear_model.LinearRegression()
        
        print('######################## Start GLM fitting #######################')
        
        regressor_coeff = [] # batch to store regressor coefficients
            
        levels = ['exp', 'id'] #fit independently for each level (e.g. experiment and subject id)
        df_temp = df.copy().reset_index().set_index(levels)
            
        index = df_temp.index.unique()
            
        for idx in tqdm( index ): #loop over each test and subject
                    
            x = df_temp[regressors[1:]].xs(idx, level=levels)  #exclude Baseline
            y = df_temp['per'].xs(idx, level=levels)
                
            model.fit(x, y) # fit regressors (except for Baseline) to predict accuracy
            regressor_coeff.append([model.intercept_] + model.coef_.tolist()) # join regressor coefficients with baseline
        
        df_regression = pd.DataFrame(columns=regressors, data=regressor_coeff, index=index) # create dataframe
        df_regression.columns.set_names('Regressor', inplace=True)
        df_regression = df_regression.stack('Regressor').unstack(['exp', 'Regressor'])
            
        #Export
        self.save.export(df_regression.stack(['exp', 'Regressor']).reset_index(), self.save.folder_regression, 'df_regression')


    #%% Master dataframe: contains key values from experiment setup
    
    def master_datafile(self, df):
            
        #Key variables
        n_trials = 20
        n_sessions = 2
            
        subjects = df.index.get_level_values('id').unique().tolist()
        tests = df.index.get_level_values('exp').unique().tolist()
    
        rw_mag = df.magnitude.unique()[0] #absolute magnitude of reward (or punishment)
        contingency = df.p1.unique()[0] #probability best option
            
        context_dict = {'RP': [contingency, rw_mag, 'partial'],
                        'RC': [contingency, rw_mag, 'complete'],
                        'PP': [1-contingency, -rw_mag, 'partial'],
                        'PC': [1-contingency, -rw_mag, 'complete']}
            
        information_dict = {'partial': -1, 'complete': 1}
        
        #Fill out master datafile       
        master_datafile = {'tests' :            tests,
                           'n_tests':           len(tests),
                           'learning_contexts': list(self.learning_contexts_dict.keys()), #learning contexts
                           'magnitude':         rw_mag,                               #reward magnitude
                           'contingency':       contingency,
                           'subjects':          subjects,
                           'n_subjects':        len(subjects),
                           'n_trials':          n_trials,
                           'n_sessions':        n_sessions,
                           'n_rounds':          len(self.learning_contexts_dict.keys())*n_sessions,
                           'context_dict':      context_dict,
                           'information_dict':  information_dict}
 
        import json
       
        file_name = 'master_datafile.json'   
 
        with open(self.save.folder_setup+file_name, 'w') as f: # An arbitrary collection of objects supported by pickle.
  
            json.dump(master_datafile, f, indent=0) #
        
        return
