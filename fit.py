#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:24:12 2020

@author: stefano
"""

#%%Import libraries

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import scipy

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from functools import partial
import multiprocessing.managers
#%% 

np.random.seed(10)

class FitRL:
    
    def __init__(self, pool_exp, model, master_datafile, export, folder, filename):
        
        #%% Optimisation process

        '''
        REINFORCEMENT LEARNING MODEL FITTING
        '''
        
        #Whether or not to pool data from two experiments (test and retest)
        self.pool_exp = pool_exp #simulations fit independently, but come from parameters fit on pooled experiments
                
        self.subjects = master_datafile['subjects']
        
        self.context_dict = master_datafile['context_dict']
    
        #%% Models of choice
        
        self.model = model
                
        self.export = export
        self.folder_parameters = folder
    
        self.filename = filename
        
        self.parameter_names = ['beta','alpha1','alpha2','alpha3', 'alpha_conf', 'alpha_disc']
        self.n_parameters = len(self.parameter_names)
                
    #%% Fit Reinforcement Learning models on data file
         
    def ML(self, df, MAP=False):

        self.MAP = MAP #maximum a posteriori, True or False to add or not prior cost to cost function
        self.subjects = [int(subject) for subject in self.subjects] #from string to int

        #Data

        # Fitting across experiments (pool_exp = True) or individual experiments
        self.df = df.reset_index().set_index('id') if self.pool_exp else df 
        #self.df_subsampling(2) # Limit to a given specified number of subjects

        self.index = self.df.index.unique()
        
        method = 'MAP' if MAP else 'ML' 
        maxbetaprecision = None if MAP else 30
                
        precision = 10**(-7)
        
        # Boundary conditions for fitting    
        self.bnds = ((0+precision, maxbetaprecision), #beta
                (0+precision, 1-precision), #alpha1
                (0+precision, 1-precision), #alpha2
                (0+precision, 1-precision), #alpha3
                (0+precision, 1-precision), #alpha confirmatory
                (0+precision, 1-precision)) #alpha disconfirmatory
        
        # Initial guess for minimisation process

        # Grid
        if self.df.reset_index()['exp'].max() > 1: #fitting simulations
            initial_guess_learning_rates = [1/3, 2/3]
            initial_guess_beta = [1, 10]
            
            if self.filename.split('_')[-1] == "extreme": #fitting simulations
                print('limited initial guess for extreme')
                initial_guess_learning_rates = [1/2]
                initial_guess_beta = [5.33] #median of beta parameter value fit in MAP/RL_parameters/df_parameters_pool_exp_True_method_MAP_model_ASYMMETRIC RELATIVE.csv'
            elif isinstance(eval(self.filename.split('_')[-1]), int): #check is integer, reflecting number of sessions simulated
                
                print('limited initial guess for n_sessions comparison')
            
                initial_guess_learning_rates = [1/2]
                initial_guess_beta = [1, 10]            
        
        else: #fitting test-retest experiments
            initial_guess_learning_rates = [0.15, 0.5, 0.85]
            initial_guess_beta = [0.5, 2, 10]
                
        self.n_guesses = len(initial_guess_learning_rates)**3 * len(initial_guess_beta) #3 learning rates + beta in ASYMMETRIC RELATIVE model
        initial_guess = [initial_guess_beta] + [initial_guess_learning_rates] * 3 #3 learning rates in ASYMMETRIC RELATIVE model            
        guesses = pd.MultiIndex.from_product( initial_guess ) 
        guesses = np.array( list(guesses) )

        self.guess_list = np.zeros((self.n_guesses, self.n_parameters))
        self.guess_list[:, 3:] = guesses[:, 1:]
        self.guess_list[:, 0] = guesses[:, 0] #beta
        
        #Pick randomly
        #self.n_guesses = 10
        #self.guess_list = np.random.rand(self.n_guesses, self.n_parameters) *(1-2*precision)+precision #set guesses between precision range
        #guess_list[:, 0] *= 5 #stretch guess for beta by multiplying random sampling  
        
        # Fit parameters for all fitting models, experiments, tests and subjects
        self.parameter_optimisation(method)

    #%%

    def df_subsampling(self, n_sbj):
        
        '''
        Subsampling from behavioural data to fit model
        '''
        
        print('\n Subsampling: ACTIVE -> subsample from subset of total subjects')

        self.subjects = list(self.df.index.get_level_values('id').unique()[0:n_sbj])
        self.df = self.df.loc[self.subjects, :] if self.pool_exp else self.df.loc[(slice(None), self.subjects), :] 

    #%% Negative log-likelihood
    
    def negative_log_likelihood(self, params, subject_data, min_cost_temp):
        
        beta, alpha1, alpha2, alpha3, alpha_conf, alpha_disc = params[0], params[1], params[2], params[3], params[4], params[5] # inputs are guesses at our parameters
        
        ll = 0 #initialise log-likelihood
        
        group_by = ['exp', 'round', 'learning context']
        subject_data = subject_data.reset_index().set_index(group_by) #group by experiment, round and learning context across tests
        subject_data.sort_index(inplace=True)
        
        #May be able to parallelise? #s.apply(lambda x: fun(x), axis=1)
        
        for (exp, rnd, context) in subject_data.index.unique():
            
            s_temp = subject_data.xs(((exp, rnd, context)), level=group_by) #temporary subject's dataframe with round-specific data
            
            _, _, information = self.context_dict[context] #context-specific information
            
            C = s_temp.choice.tolist() #choice
            R = s_temp.outcome.tolist() #outcome: reward/punishment
            Ru = s_temp.outcome_unchosen.tolist() #unchosen reward/punishment
    
            Q = np.zeros((2))       #temporary batch for q value update
            V = 0                   #initialise value for RELATIVE model  
                
            for t in s_temp.trial:
                
                if -ll > min_cost_temp: #if cost is already above min cost, skip computations
                
                    break
                
                else: #otherwise, keep accumulating
                                 
                    log_p = beta*Q[ C[t] ]                  #log of probability of choosing chosen option C (0 or 1)
                    log_p -= scipy.special.logsumexp(beta*Q) #normalise by log of sum of probabilities sum(exp^(-beta*Q_i))
                
                    ll += log_p #sum log of probabilities to compute log-likelihood
                   
                    #choice
                    c = C[t] #chosen
                    u = 1-C[t] #unchosen
                    
                    ### Relative model (e.g. Relative and asymmetric relative) 
            
                    if self.model == 'ASYMMETRIC RELATIVE': #Palminteri et al. 2015
                                        
                        # !!! q[u] is used BEFORE being updated
                        r_v = (R[t] + Q[u])/2 if information == 'partial' else (R[t] + Ru[t])/2 if information == 'complete' else None
                        #if information is partial                                              #if information is complete
                        #estimate reward as mean between absolute reward and action value       #estimate reward as mean between absolute reward and unchosen reward
                    
                        #Delta
                        delta_V = r_v - V
                    
                        V += alpha3 * delta_V #estimate value from r_V, updating for next time, BEFORE using it to update Q

                        #Delta
                        delta_c = R[t] -V - Q[c] #factual
                        delta_u = Ru[t] -V - Q[u] #counterfactual
    
                    
                        ### Asymmetry
                                            
                        Q[c] += alpha_conf * delta_c if delta_c > 0 else alpha_disc * delta_c
                        
                        if information=='complete':
                            Q[u] += alpha_disc * delta_u if delta_u > 0 else alpha_conf * delta_u
                                
        return -ll  
    
    #%% Define a likelihood-based cost function
    
    def prior_cost(self, params):
    
        beta, alpha1, alpha2, alpha3, alpha_conf, alpha_disc = params[0], params[1], params[2], params[3], params[4], params[5] # inputs are guesses at our parameters
    
        #Priors
        
        #Suggestion taken from https://www.princeton.edu/~ndaw/d10.pdf
        
        # one possible alternative to a hard constraint on parameters is a prior.
        # In particular, equation 1 suggests that prior information about
        # the likely range of the parameters could enter via the term P(θM|M), and would serve to regularize the
        # estimates. In this case we would use a maximum a posteriori estimator for ˆθM: i.e., optimize the (log) product
        # of both terms on the right hand side of Equation 1, rather than only the likelihood function
        
        # Values of prior distributions from Palminteri et al., 2015), https://www.nature.com/articles/ncomms9096
        # " P(θn) is calculated based on the parameters value retrieved from the parameter optimization procedure,
        # assuming learning rates beta distributed (betapdf(parameter,1.1,1.1))
        # and softmax temperature gamma-distributed (gampdf(parameter,1.2,5)) [68].
        # The present distributions have been chosen to be relatively flat
        # over the range of parameters retrieved in the previous and present studies."
        # [68] Worbe et al., 2016 (https://www.nature.com/articles/mp201546)
        
        prior_beta = scipy.stats.gamma.pdf(beta, a=1.2, scale=5)
        
        cost = - np.log(prior_beta)
            
        if 'ASYMMETRIC' in self.model:
            prior_alpha_conf = scipy.stats.beta.pdf(alpha_conf, 1.1, 1.1) #same distribution as alpha1?
            cost -= np.log(prior_alpha_conf)
            
            prior_alpha_disc = scipy.stats.beta.pdf(alpha_disc, 1.1, 1.1) #same distribution as alpha1?
            cost -= np.log(prior_alpha_disc)
    
        elif 'ASYMMETRIC' not in self.model: #(e.g. absolute or 'pure' RELATIVE model)
            prior_alpha1 = scipy.stats.beta.pdf(alpha1, 1.1, 1.1) #cdf-cdf?
            cost -= np.log(prior_alpha1)
            
            prior_alpha2 = scipy.stats.beta.pdf(alpha2, 1.1, 1.1) #same distribution as alpha1?
            cost -= np.log(prior_alpha2)
    
        if 'RELATIVE' in self.model:
            prior_alpha3 = scipy.stats.beta.pdf(alpha3, 1.1, 1.1) #same distribution as alpha1?
            cost -= np.log(prior_alpha3)   
            
        return cost
    
    def cost_function(self, params, subject_data, min_cost_temp):
        
        cost = self.prior_cost(params) * self.MAP #0 if ML, prior if MAP
        cost += self.negative_log_likelihood(params, subject_data, min_cost_temp)
        
        return cost

    def fit_subject(self, parameters_optimal, idx_tuple):
        
        idx_n, idx = idx_tuple
        
        print(idx_n, idx)
        
        subject_data = self.df.loc[idx] #subject's dataframe #deprecated, it slows down computation
        #subject_data = self.df.xs(idx, level=self.index.names, drop_level=False) #subject's dataframe
        
        cost = np.zeros((len(self.index), self.n_guesses))
        parameters = np.zeros((len(self.index), self.n_guesses, self.n_parameters))
        
        # L-BFGS-B algorithm as optimisation method from https://conference.scipy.org/proceedings/scipy2016/pdfs/alejandro_weinstein.pdf
        # in turn from https://digital.library.unt.edu/ark:/67531/metadc666315/m2/1/high_res_d/204262.pdf .
        # You can also change "maxlsint" as optional input, which indicates the maximum number of line search steps (per iteration). Default is 20.
                
        for guess_n, guess in enumerate( self.guess_list ): #Loop over initial guesses for gradient descent
            
            if 'RELATIVE' not in self.model:
                guess[3] *= 0 #set alpha3 to 0 for ABSOLUTE and ASYMMETRIC model

            if 'ASYMMETRIC' not in self.model:
                guess[4] *= 0 #set alpha_conf to 0 for ABSOLUTE model
                guess[5] *= 0 #set alpha_disc to 0 for ABSOLUTE model
                
            elif 'ASYMMETRIC' in self.model:
                guess[1] *= 0 #set alpha1 to 0 for ASYMMETRIC models
                guess[2] *= 0 #set alpha2 to 0 for ASYMMETRIC models
            
            # !!! minimise NLL
            #nll = minimize(self.negative_log_likelihood, guess, args=(model, s), method='L-BFGS-B', bounds=self.bnds)
            #NLL_temp.append(nll.fun) #nll
            
            #print(f'Guess {guess_n} beta: {guess[0]}')
            
            min_cost_temp = min( cost[idx_n, :guess_n] ) if guess_n > 0 else np.inf #maximal cost seen, to make computations more efficient
            
            #Minimise cost-function to fit parameters
            res = minimize(self.cost_function, guess, args=(subject_data, min_cost_temp), method='L-BFGS-B', bounds=self.bnds)
            parameters[idx_n, guess_n] = res.x  #fitted parameters from minimised cost function
            cost[idx_n, guess_n] = res.fun #cost
        
        best_fit = np.argmin(cost[idx_n]) #get parameters where cost is minimum
        parameters_optimal[idx_n] = parameters[idx_n, best_fit] 
    
    
    def optimise_in_loop(self):
        
        # Batches to store fitted parameters and cost            

        parameters_optimal = np.zeros((len(self.index), self.n_parameters))
        
        [self.fit_subject(parameters_optimal, idx_tuple) for idx_tuple in enumerate(self.index)]
        
        return parameters_optimal
    
    
    def optimise_in_parallel(self):
                
        class MyManager(multiprocessing.managers.BaseManager):
            pass
        
        MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
        
        m = MyManager()
        m.start()
        
        parameters_optimal = m.np_zeros((len(self.index), self.n_parameters))
        
        n_cpu = multiprocessing.cpu_count()-1
        pool = Pool(n_cpu)
        
        func = partial(self.fit_subject, parameters_optimal)
        _ = pool.map(func, enumerate(self.index))
        
        return np.array( parameters_optimal )
    
        
    def parameter_optimisation(self, method):
        
        print('\n ##################### Start RL model fitting #####################') #just for the purpose of plotting lines for terminal
        print('\n Pool experiments? {0}'.format(self.pool_exp))
        print('\n Computational model to be fitted: ', self.model)
        print('\n Method: ', method)
        print(self.guess_list)
        
        #parameters_optimal = self.optimise_in_loop()
        parameters_optimal = self.optimise_in_parallel()
        
        self.create_df_parameters(parameters_optimal)
        
    def create_df_parameters(self, parameters):

        #Parameters' dataframe for all subjects in a given experiment configuration (test, retest or pooled)
        df_parameters = pd.DataFrame(parameters, columns = self.parameter_names, index=self.index)
        df_parameters.index.name = 'id'
            
        # Export                                
        print('\n Saving file... \n'), self.export(df_parameters.reset_index(), self.folder_parameters, self.filename)
        print('Done')
        
        if not self.pool_exp and self.df.reset_index()['exp'].max() == 1: #test-rest fit on experiments
            data = df_parameters['beta'].max()
            max_beta = pd.DataFrame(data=[data], columns=['beta'])
            self.export(max_beta, self.folder_parameters, 'max_beta')

    def df_parameters_shuffled(self, df_parameters, pool_exp, method, model):
        
        '''
        Shuffle vertically, between-subjects within-variable (column)
        '''
        
        #Parameters
        df_parameters = df_parameters.apply(lambda x: x.sample(frac=1).values, axis=1) 
        
        #Export
        self.export(df_parameters.reset_index(), self.folder_parameters, f'df_parameters_pool_exp_{pool_exp}_method_{method}_model_{model}_shuffled.csv')
    
    def df_parameters_synthetic(self):
        
        self.index = self.subjects #[:10] #subsampling for diagnostics
        N = len(self.index)       
        parameters = np.random.rand(N, self.n_parameters)
            
        self.create_df_parameters(parameters)
        
        return

