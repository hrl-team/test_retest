#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:59:43 2023

@author: stefanovrizzi
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

#%% Simulations

class Simulations:
    
    def __init__(self, export, folder, master_datafile, pool_exp, model, method, real_seq, shuffled, suffix, n_sessions=None, extreme=False):
    
        self.folder = folder
        self.export = export
        self.learning_contexts = master_datafile['learning_contexts']        
        self.n_sessions = master_datafile['n_sessions'] if not n_sessions else n_sessions
        self.n_trials = master_datafile['n_trials']
        self.context_dict = master_datafile['context_dict']
        self.pool_exp = pool_exp

        self.model = model
        self.method = method
        self.real_seq = real_seq if not extreme else False
        
        self.extreme = extreme #softmax to generate data, with p=1 for best option

        self.filename_simulations = 'df_simulations'+suffix
        self.filename_accuracy = 'df_accuracy_simulations'+suffix
    
    def simulate_in_parallel(self, index):

        n_cpu = mp.cpu_count()-1
        pool = mp.Pool(processes=n_cpu)
        manager = mp.Manager()

        data = manager.list()

        [pool.apply_async(self.simulate_subject, args=[idx, data]) for idx in index]

        pool.close()
        pool.join()
        
        data = np.array(data)

        return data

    def df_simulations(self, df_parameters, df):
        
        self.df_parameters = df_parameters
        self.df = df
    
        self.n_tests = 200
    
        self.round_context = self.learning_contexts * self.n_sessions #batch of learning contexts for all rounds

        index = self.df_parameters.index.unique()    

        #data = []        
        #[ self.simulate_subject(idx, data) for idx in index ] #for each model, subject
        
        data = self.simulate_in_parallel(index)    

        #General columns to be used for multi-indexing
        index = ['exp', 'id']
        
        #Task-specific columns
        columns = ['session','round','learning context','trial',
                   'choice',
                   'outcome','outcome_unchosen','outcome_0','outcome_1',
                   'Q0','Q1','V', 'valence', 'information', 'interaction']
        
        df_simulations = pd.DataFrame(data=data, columns=index+columns)

        float_columns = ['outcome', 'outcome_unchosen', 'outcome_1', 'Q0', 'Q1', 'V', 'valence', 'information', 'interaction']
        df_simulations[float_columns]= df_simulations[float_columns].astype(float)
        int_columns = ['session', 'round', 'trial', 'choice', 'valence', 'information', 'interaction']
        df_simulations[int_columns]= df_simulations[int_columns].astype(int)

        df_simulations['per']= (df_simulations.choice-.5)*2 # -1/1 for suboptimal/optimal as dummy variable for GLM
        
        df_simulations.set_index(index, inplace=True)
        df_simulations.sort_index(inplace=True)
    
        ### Export dataframe in folder        
        print('Saving file in simulations folder...')
        self.export(df_simulations.reset_index(), self.folder, self.filename_simulations)                                                                         
        print('Done')
        
    def simulate_subject(self, idx, data):

        print(idx, len(data))

        #get parameters from fitted dataframe
        beta = self.df_parameters.xs(idx)['beta']
        alpha1 = self.df_parameters.xs(idx)['alpha1']
        alpha2 = self.df_parameters.xs(idx)['alpha2']
        alpha3 = self.df_parameters.xs(idx)['alpha3']
        alpha_conf = self.df_parameters.xs(idx)['alpha_conf']
        alpha_disc = self.df_parameters.xs(idx)['alpha_disc']

        parameters = [beta, alpha1, alpha2, alpha3, alpha_conf, alpha_disc]
            
        [self.perform_task(test, data, idx, parameters) for test in range(self.n_tests)] #for each test

    def df_accuracy(self, df):

        '''
        Compute accuracy dataframe from simulations
        '''
        
        #Compute accuracy
        df_accuracy = df.groupby(['exp', 'learning context', 'id'])['choice'].mean().rename('accuracy')
                
        self.export(df_accuracy.reset_index(), self.folder, self.filename_accuracy)
        
        return df_accuracy

    #%% Model-dependent Q-algorithm to run simulations
    
    def Q_learning(self, c, R, Q, V, information, learning_rates):

        [alpha1, alpha2, alpha3, alpha_conf, alpha_disc] = learning_rates
        
        u = 1-c #unchosen
    
        #Delta
        delta_c = R[c] - Q[c] #factual
        delta_u = R[u] - Q[u] #counterfactual
    
        ### ABSOLUTE model
        
        if self.model=='ABSOLUTE': 
            
            Q[c] += alpha1 * delta_c
            Q[u] += alpha2 * delta_u if self.information=='complete' else 0
            
        ### Relative models    
            
        if 'RELATIVE' in self.model:
            
            #X_star = 0
            X_star = Q[u]
            #X_star = #paired outcome?
            #X_star = #last seen outcome associated with the option
            
            r_v = (R[c] + X_star)/2 if information == 'partial' else (R[c] + R[u])/2 if information == 'complete' else None
            #estimate reward as mean between absolute reward        #estimate reward as mean
            #and action value                                       #between absolute reward and unchosen reward
                   
            #Delta V
            delta_V = r_v - V
            V += alpha3 * delta_V #estimate value from r_V, updating for next time, BEFORE using it to update Q

            #Delta Q
            delta_c -= V
            delta_u -= V
                        
        ### ASYMMETRIC model
            
        if 'ASYMMETRIC' in self.model:
            
            Q[c] += alpha_conf * delta_c if delta_c > 0 else alpha_disc * delta_c
                        
            if information=='complete':
                Q[u] += alpha_disc * delta_u if delta_u > 0 else alpha_conf * delta_u  
        
        ### RELATIVE model
     
        elif self.model=='RELATIVE':
                            
            Q[c] += alpha1 * delta_c
            Q[u] += alpha2 * delta_u if information=='complete' else 0
    
        return Q, V

    # Task
    
    def perform_task(self, test, data, idx, parameters):
        
        [beta, alpha1, alpha2, alpha3, alpha_conf, alpha_disc] = parameters
        
        learning_rates = [alpha1, alpha2, alpha3, alpha_conf, alpha_disc]

        #Specify task
        for rnd, context in enumerate(self.round_context):
            
            session = int( rnd/len(self.context_dict.keys()) ) #session number
            
            p_best_option, rw, information = self.context_dict[context] #get context-related values
            p_best_option = round(p_best_option) if self.extreme else p_best_option #round to 0 or 1 probabilities to adopt extreme case
            
            information_dummy = ((information=='complete')-.5)*2 #transform string into -1 / +1 for 'partial' / 'complete' information
            
            valence = np.sign(rw) #np.sign(rw)[0]
            interaction = valence*information_dummy
            
            #Initialise values for this round            
            Q = np.zeros((2))
            V = 0
            
            R = self.get_reward_seq(test, p_best_option, rw, session, context, idx) #set rewards
            
            for t in range(self.n_trials): #run simulations
                                
                C = self.SDR(Q, beta) #choice from Soft-max Decision Rule            
                Q, V = self.Q_learning(C, R[t, :], Q, V, information, learning_rates)
    
                data.append([test, idx, session, rnd, context, t, C, R[t, C], R[t, 1-C], R[t, 0], R[t, 1], Q[0], Q[1], V, valence, information_dummy, interaction])
    
    # Softmax Decision Rule
    
    def SDR(self, Q, beta): 
        
        p = 1 / (1 + np.exp( (Q[0] - Q[1]) * beta ) ) #probability of sampling option 1
        c = np.random.choice([0, 1], p=[1-p, p]) #sample choice from given distribution
        
        return c    
    
    # Extract reward sequence
    
    def real_seq_(self, p_best_option):
                
        reward = self.df.reset_index().set_index(['exp', 'session', 'learning context', 'id', 'trial'])
        reward = reward[['choice', 'outcome', 'outcome_unchosen']]
        
        reward['option_0'] = (reward.choice==1)*(reward.outcome_unchosen)+(reward.choice==0)*(reward.outcome) #assign to option 0 outcomes from options 0
        reward['option_1'] = (reward.choice==0)*(reward.outcome_unchosen)+(reward.choice==1)*(reward.outcome) #assign to option 1 outcomes from options 1
        
        reward.drop(['choice', 'outcome', 'outcome_unchosen'], axis=1, inplace=True)
        
        idxe = pd.IndexSlice
        
        #Reward
        lc = 'RP'
        index = reward[reward.isna()].loc[idxe[:, :, [lc], :]].index
        mask_RP = pd.DataFrame(index=index)
        
        option = 'option_0'
        idx = reward[reward[option].isna()][option].loc[idxe[:, :, [lc], :]]
        mask_RP[option] = idx.apply(lambda x: .5* (np.random.rand() > p_best_option) )
        
        option = 'option_1'
        idx = reward[reward[option].isna()].loc[idxe[:, :, [lc], :]][option]
        mask_RP[option] = idx.apply(lambda x: .5* (np.random.rand() < p_best_option) )
        
        #Punishment
        lc = 'PP'
        index = reward[reward.isna()].loc[idxe[:, :, [lc], :]].index
        mask_PP = pd.DataFrame(index=index)
        
        option = 'option_0'
        idx = reward[reward[option].isna()][option].loc[idxe[:, :, [lc], :]]
        mask_PP[option] = idx.apply(lambda x: -.5* (np.random.rand() < p_best_option) )
        
        option = 'option_1'
        idx = reward[reward[option].isna()][option].loc[idxe[:, :, [lc], :]]
        mask_PP[option] = idx.apply(lambda x: -.5* (np.random.rand() > p_best_option) )
        
        reward = reward.combine_first(mask_RP) #replace NaN with sampled rewards and punishments
        reward = reward.combine_first(mask_PP)
        
        #check successful fill-in
        #print(reward.groupby(['exp', 'session', 'learning context']).mean()/0.5) #(0.5 is magnitude, divide by it to get probabilities. Option 1 is optimal option
        #print(reward.groupby(['exp', 'session', 'learning context']).apply(lambda x: x.isna()).sum()) #check no NaN left
        
        return reward

    def get_reward_seq(self, test, p_best_option, rw, session, context, idx):
        
        R = np.zeros((self.n_trials, 2)) # batch for rewards for both options
        
        if self.real_seq: #get outcomes from real experiments
                    
            reward = self.real_seq_(p_best_option)
            reward = reward.sort_index() #may increase performance in slicing dataframe
            
            R[:, 0] = reward['option_0'].loc[pd.IndexSlice[(test % 2, session, context, idx)]] # % 2 = mod 2, to alternate test and retest
            R[:, 1] = reward['option_1'].loc[pd.IndexSlice[(test % 2, session, context, idx)]]
            
        elif not self.real_seq: #generate random sequence by sampling rewards for the whole round and both options: 0/1 is suboptimal/optimal
                    
            if rw > 0: #Reward
                R[:, 0] = ( np.random.rand(self.n_trials) > p_best_option ) #rewards for option 0
                R[:, 1] = ( np.random.rand(self.n_trials) <= p_best_option ) #rewards for option 1

            elif rw < 0: #Punishment, p_best_option is 1-p_best_option from reward (see "context_dict" within df_master_() function in hood_analysis.py)
                R[:, 0] = ( np.random.rand(self.n_trials) > p_best_option ) #rewards for option 0
                R[:, 1] = ( np.random.rand(self.n_trials) <= p_best_option ) #rewards for option 1
                
            R *= rw #scale to magnitude and sign

        return R
