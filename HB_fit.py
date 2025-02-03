#import pystan as stan
import stan
#import stan_utility
import pandas as pd
import numpy as np
import arviz as az
import os.path

from pystan_model import model_reduced, model_ind, model_independent_normal
import matplotlib.pyplot as plt
import seaborn as sns

class HierarchicalBayesian:

  def __init__(self, df, method, pool_exp, hbModelType, folder_parameters, export):

    self.df = df
    self.experiments = ['test', 'retest'] if method == 'HB' else [None] if method == 'HBpool' else None
    self.pool_exp = pool_exp
  
    self.hbModelType = hbModelType
    model_dict = {'independent_normal': model_independent_normal,
                  'independent': model_ind,
                  'reduced': model_reduced}
  
    self.model = model_dict[hbModelType]
    self.method = method

    self.parameterNames = ['alpha_conf', 'alpha_disc', 'alpha_v', 'beta_par']
    self.Nparameters = len(self.parameterNames)  
    
    self.folder_parameters = folder_parameters

    self.export = export
    
  def fit(self):   
  
    for experiment in self.experiments:
      
      self.df_summary_filename = f'df_summary_{self.method}_{experiment}'
      self.df_fit_filename = f'df_fit_{self.method}_{experiment}'
          
      fname_summary = os.path.join(self.folder_parameters, self.df_summary_filename+'.csv')
      fname_fit = os.path.join(self.folder_parameters, self.df_fit_filename+'.csv')

      if not os.path.isfile(fname_summary) * os.path.isfile(fname_fit): #if files have not been saved, fit

        self.prepare_data(experiment)
        self.fit_model()
        #self.plot_diagnostics()
        self.export_files(experiment)
    
  def prepare_data(self, experiment):

    df = self.df.reset_index()
    df['exp_str'] = df['exp'].map({0: 'test', 1: 'retest'})
    df = df[df['exp_str'] == experiment] if experiment else df #fit experiment data independently ('test'/'retest') or jointly (None)

    num_exp_samples = 1 if self.pool_exp else len(df['exp'].unique())

    df['information'] = (df['information'] / 2 + 0.5).astype(int)  
    df.outcome_unchosen = df.outcome_unchosen.fillna(0)
    
    df.sort_values(by=['id', 'exp', 'round', 'trial'], inplace=True)
    #df = df[df['id'].isin(df['id'].unique()[:3])] #subsample 3 subjects, for diagnostics

    subjects = df.id.apply(lambda x: int(x)).unique()
    
    num_subjects = len(subjects)
    num_experiments = len(df['exp'].unique())
    num_rounds = df['round'].max()
    trials_per_round = df['trial'].max()+1

    # Prepare data for PyStan
    self.data = {
        'num_experiments': num_experiments,
        'num_exp_samples': num_exp_samples,       
        'num_subjects': num_subjects,
        'num_rounds': num_rounds,
        'num_trials': trials_per_round,
        'Outcome': np.reshape(df['outcome'].values, (num_subjects, num_experiments, num_rounds, trials_per_round)),
        'OutcomeUnchosen': np.reshape(df['outcome_unchosen'].values, (num_subjects, num_experiments, num_rounds, trials_per_round)),
        'Information': np.reshape(df['information'].values, (num_subjects, num_experiments, num_rounds, trials_per_round)),
        'Action': np.reshape(df['choice'].values, (num_subjects, num_experiments, num_rounds, trials_per_round)),
        'P': self.Nparameters
    }

  def fit_model(self):

    # Compile the model
    self.posterior = stan.build(self.model, data=self.data)

    # Fit the model to the data
    self.fit = self.posterior.sample(num_chains=4, num_samples=1000, num_warmup=1000) #num_warmup=100
    
    #stan_utility.check_n_eff(self.fit)
    #stan_utility.check_rhat(self.fit)
    #stan_utility.check_treedepth(self.fit)
    #stan_utility.check_energy(self.fit)
    #stan_utility.check_div(self.fit)
    #stan_utility.check_all_diagnostics(self.fit)
        
  def plot_diagnostics(self, experiment):

    # Visualize the results using arviz

    if self.hbModelType == 'reduced':
    
      population_alphas = ['a_population_alpha_conf', 'b_population_alpha_conf',
                            'a_population_alpha_disc', 'b_population_alpha_disc',
                            'a_population_alpha_v', 'b_population_alpha_v']
    
      population_beta = ['a_population_beta', 'b_population_beta']
    
      individual_alphas = ['alpha_conf', 'alpha_disc', 'alpha_v']
      individual_beta = ['beta_par']

      var_names_dict = {'population_alphas': population_alphas,
                      'population_beta': population_beta,
                      'individual_alphas': individual_alphas,
                      'individual_beta': individual_beta}
                      
    elif self.hbModelType == 'independent':
    
      population_alphas_a = ['a_a_population_alpha_conf', 'b_a_population_alpha_conf',
                            'a_a_population_alpha_disc', 'b_a_population_alpha_disc',
                            'a_a_population_alpha_v', 'b_a_population_alpha_v']
    
      population_beta_a = ['a_a_population_beta', 'b_a_population_beta']
    
      population_alphas_b = ['a_b_population_alpha_conf', 'b_b_population_alpha_conf',
                            'a_b_population_alpha_disc', 'b_b_population_alpha_disc',
                            'a_b_population_alpha_v', 'b_b_population_alpha_v']
    
      population_beta_b = ['a_b_population_beta', 'b_b_population_beta']
    
      individual_alphas_hyper = ['a_subject_alpha_conf', 'b_subject_alpha_conf',
                            'a_subject_alpha_disc', 'b_subject_alpha_disc',
                            'a_subject_alpha_v', 'b_subject_alpha_v']
    
      individual_beta_hyper = ['a_subject_beta', 'b_subject_beta']
    
      individual_alphas = ['alpha_conf', 'alpha_disc', 'alpha_v']
      individual_beta = ['beta_par']

      var_names_dict = {'population_alphas_a': population_alphas_a,
                      'population_beta_a': population_beta_a,
                      'population_alphas_b': population_alphas_b,
                      'population_beta_b': population_beta_b,
                      'individual_alphas_hyper': individual_alphas_hyper,
                      'individual_beta_hyper': individual_beta_hyper,
                      'individual_alphas': individual_alphas,
                      'individual_beta': individual_beta}

    elif self.hbModelType == 'independent_normal':
    
      population_sigma = ['sigma_pr']
      individual_mu = ['mu_pr_sub']
      individual_sigma_r = ['sigma_pr_r']

      individual_alphas_pr = ['alpha_conf_pr', 'alpha_disc_pr', 'alpha_v_pr']
      individual_beta_pr = ['beta_par_pr']
    
      individual_alphas = ['alpha_conf', 'alpha_disc', 'alpha_v']
      individual_beta = ['beta_par']

      var_names_dict = {'population_sigma': population_sigma,
                       'individual_mu': individual_mu,
                       'individual_sigma_r': individual_sigma_r,
                      'individual_alphas_pr': individual_alphas_pr,
                      'individual_beta_pr': individual_beta_pr,
                      'individual_alphas': individual_alphas,
                      'individual_beta': individual_beta}

    for var_names_group, var_names in var_names_dict.items():

      az.plot_trace(self.fit, var_names=var_names), plt.tight_layout(), self.save.figure(f'traces_{var_names_group}_{experiment}', HB=True)
      az.plot_posterior(self.fit, var_names=var_names, backend_kwargs={'sharex': True}), self.save.figure(f'posteriors_{var_names_group}_{experiment}', HB=True)
      #az.plot_forest(self.fit, var_names=var_names), plt.tight_layout(), self.save.figure(f'forests_{var_names_group}_{experiment}', HB=True)

    az.plot_pair(self.fit, divergences=True), self.save.figure(f'divergences_method_{self.method}_exp_{experiment}', HB=True)

    #self.ridgeline_plot(self.fit.to_frame())

  def export_files(self, experiment):

    df_summary = az.summary(self.fit)
    self.export(df_summary.reset_index(), self.folder_parameters, self.df_summary_filename)
    
    df_fit = self.fit.to_frame()
    self.export(df_fit.reset_index(), self.folder_parameters, self.df_fit_filename)
    
    df_fit_mean = df_fit.mean().to_frame() #collapse to mean of posterior (over all chains) as point estimate

    self.export(df_fit_mean.reset_index(), self.folder_parameters, f'df_fit_{self.method}_mean_{experiment}') 
    
    self.check_diagnostics(df_fit, df_summary, experiment)

  def check_diagnostics(self, df_fit, df_summary, experiment):

    diagnostics = {'r_hat_mean': df_summary.r_hat.mean(),
    'r_hat_min': df_summary.r_hat.min(),
    'r_hat_max': df_summary.r_hat.max(),
    'ess_tail_mean': df_summary.ess_tail.mean(),
    'ess_tail_min': df_summary.ess_tail.min(),
    'ess_tail_max': df_summary.ess_tail.max(),
    'ess_bulk_mean': df_summary.ess_bulk.mean(),
    'ess_bulk_min': df_summary.ess_bulk.min(),
    'ess_bulk_max': df_summary.ess_bulk.max(),    
    'divergence': df_fit.divergent__.sum(),
    'max_treedepth': df_fit.treedepth__.max(),
    'E-BFMI': ((df_fit.energy__.diff().iloc[1:])**2).sum() / len(df_fit.energy__) / df_fit.energy__.var()
    }

    df_diagnostics = pd.DataFrame.from_dict(diagnostics, orient='index', columns=['Value'])
    self.export(df_diagnostics.reset_index(), self.folder_parameters, f'df_diagnostics_{self.method}_exp_{experiment}')

  #def ridgeline_plot(self, df):
    
  #  for parameter in self.parameterNames:

  #    plt.figure(figsize=(25, 6))
  #    sns.stripplot(data=df, y=parameter, x='rank', jitter=False, alpha=.05, palette=['b', 'r'], legend=False, hue='exp') #, dodge=True)
  #    self.save.figure(f'HB_violin_{parameter}', HB=True)
