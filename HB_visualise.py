import pymc as pm
import numpy as np

import arviz as az
import pymc as pm

#import pymc.sampling.jax as pmjax

import pytensor
import pytensor.tensor as pt
import pandas as pd

az.style.use("arviz-darkgrid")

from filemanager import Fetch, Save

save = Save()
fetch = Fetch(save)
        
df = fetch.df()
df = df.reset_index()
df['exp'] = df['exp'].map({0: 'test', 1: 'retest'})
experiment = 'test'
df = df[df['exp']==experiment]
df.information = (df.information/2+.5).astype(int) #0 and 1 instead of -1 and 1

num_subjects = len(df.id.unique())

# PyMC3 model
with pm.Model() as hierarchical_model_pymc:

    # Population-level parameters
    a_population_alpha_conf = pm.Gamma('a_population_alpha_conf', alpha=5, beta=1)
    a_population_alpha_disc = pm.Gamma('a_population_alpha_disc', alpha=5, beta=1)
    a_population_alpha_v = pm.Gamma('a_population_alpha_v', alpha=5, beta=1)
    a_population_beta = pm.Gamma('a_population_beta', alpha=5, beta=1)

    b_population_alpha_conf = pm.Gamma('b_population_alpha_conf', alpha=5, beta=1)
    b_population_alpha_disc = pm.Gamma('b_population_alpha_disc', alpha=5, beta=1)
    b_population_alpha_v = pm.Gamma('b_population_alpha_v', alpha=5, beta=1)
    b_population_beta = pm.Gamma('b_population_beta', alpha=5, beta=1)

    # Individual-level parameters
    alpha_conf = pm.Beta('alpha_conf', alpha=a_population_alpha_conf, beta=b_population_alpha_conf, shape=num_subjects)
    alpha_disc = pm.Beta('alpha_disc', alpha=a_population_alpha_disc, beta=b_population_alpha_disc, shape=num_subjects)
    alpha_v = pm.Beta('alpha_v', alpha=a_population_alpha_v, beta=b_population_alpha_v, shape=num_subjects)
    beta = pm.Gamma('beta', alpha=a_population_beta, beta=b_population_beta, shape=num_subjects)
                             
    parameters = {'alphaconf': alpha_conf,
                  'alphadisc': alpha_disc,
                  'alphav': alpha_v,
                  'beta': beta}

    #for name in df.id.unique():
            
    #    df_temp = df[df['id']==name]
                                        
    #    like = pm.Potential(name=f"{name}_{experiment}", var=total_loglikelihood(parameters, df_temp))


graph = pm.model_to_graphviz(model=hierarchical_model_pymc)
graph.render(filename=f'Figures/hierarchical_model_structure_{experiment}', format='png', cleanup=True)

#import matplotlib.pyplot as plt
#plt.show()
#save.figure(f'model_{experiment}')