#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:01:05 2022

@author: svrizzi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import string
#import matplotlib.gridspec as gs
from filemanager import Save, Fetch

from scipy.stats import ttest_rel

import matplotlib.gridspec as gridspec

#%%

class Colors:
    
    def __init__(self):
            
        self.learning_contexts = dict({'RP': [0, .5, 0],   #RP
                                   'PP': [.5, 0, 0],   #PP
                                   'RC': [0, .5, .5],  #RC
                                   'PC': [.5, 0, .5]}) #PC

        self.grey = [.5, .5, .5]
        self.grey_4 = [self.grey]*4
        
        self.palette = [self.grey, 'k']
        
        self.double = [color for color in self.learning_contexts.values()]*2 #to plot test-retest violin plots
        self.double = [y for x in self.learning_contexts for y in self.double if y == x] #sort according to learning context order

class Plot:
    
    def __init__(self, folder):

        self.save = Save(folder)
        self.fetch = Fetch(self.save)
        self.colors = Colors()
        
        self.learning_contexts = self.fetch.master_datafile()['learning_contexts']
        
        ##############################
                
        #Font
        plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.sf'] = 'Arial'
        
        plt.rcParams["axes.labelweight"] = "bold"
        
        #remove top & right border in any plot
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        ##############################
        
        self.exp_dict = {0: 'test', 1: 'retest'} #retest comes alphabetically first, before test: Be careful about sorting and plotting
        self.hue_order_exp = self.exp_dict.values()

        self.method_dict = dict({'ML': r'$ML$',
                            'MAP': r'$MAP$',
                            'HB': r'$HB$',
                            'HBpool': r'$HB_{pool}$'})
        
        self.learning_contexts_verbose = ['Reward Partial', 'Punishment Partial', 'Reward Complete', 'Punishment Complete']      
        self.learning_contexts_dict = dict(zip( self.learning_contexts, self.learning_contexts_verbose) )

        self.point_size = 10
        
        #Adjust greek letter to no italics
        self.labels_to_replace = {r'$\alpha_{v}$': r"$\mathsf{\alpha_{\mathsf{V}}}$",
                                  r'$\alpha_{CON}$': r"$\mathsf{\alpha_{\mathsf{CON}}}$",
                                  r'$\alpha_{DISC}$': r"$\mathsf{\alpha_{\mathsf{DISC}}}$",
                                  r'$\beta$': r"$\mathsf{\beta}$"}    

class Figure2(Plot):
    
    def __init__(self, reliability_measure, folder):
        
        super().__init__(folder)
        
        self.reliability_measure = reliability_measure

    def plot_pca(self, df_name, method=None, join_plots=False):
                
        df_scores_test = self.fetch.df_scores(exp=0, df_name=df_name, method=method)
        df_scores_test['exp'] = 0

        df_scores_retest = self.fetch.df_scores(exp=1, df_name=df_name, method=method)
        df_scores_retest['exp'] = 1

        df_scores = pd.concat([df_scores_test, df_scores_retest])

        df_scores.reset_index(inplace=True)
        #df_scores = df_scores.reset_index().set_index(['exp', 'id']).drop('index', axis=1).stack().to_frame().rename(columns={0:'Scores'}).reset_index().rename(columns={'level_2':'PC'})

        df_scores['exp'] = df_scores['exp'].map(self.exp_dict)
        df_scores = df_scores.reset_index().set_index(['exp', 'id']).drop('index', axis=1)

        num_pcs = df_scores.shape[1]

        self.join_plots = join_plots

        ####### Plots' functions

        def plot_loadings(pc_n, pc):

                ax = self.fig.add_subplot(gs[0, pc_n]) if self.join_plots else self.fig.add_subplot(gs[0, 0])

                data = df_scores[pc].unstack('exp')
                
                sign_correction = np.sign( df_reliability.loc[pc][reliability_measure] )
                df_reliability.loc[pc, reliability_measure] = df_reliability.loc[pc, reliability_measure] * sign_correction #CORRECTING TO POSITIVE CORRELATION
                data['retest'] *= sign_correction # CORRECTING SIGN OF RETEST VALUES TO HAVE POSITIVE CORRELATION

                g0 = sns.regplot(x="test", y="retest",
                            data=data,
                            color=self.colors.grey,
                            scatter_kws={'s': self.point_size},
                            line_kws={'color': 'k', 'lw': 2},
                            ax = ax)
        
                g0.set_title(pc)
                g0.set_xlabel('PC score (test)')
                correctiontext = '\n [RETEST SIGN HAS BEEN FLIPPED]' if sign_correction < 0 else ''
                if self.join_plots:
                    g0.set_ylabel(f'PC score (retest)'+correctiontext) if pc_n==0 and self.join_plots else g0.set_ylabel(correctiontext) #g0.set_ylabel('')
                else:
                    g0.set_ylabel(f'PC score (retest)'+correctiontext)
                
                self.ax_limits(pc_n, pc, ax)

                # Loop to write R and draw a line of x=y
                self.text_reliability(reliability_measure, pc, df_reliability, g0)
                
                # Draw a line of x=y 
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                g0.plot(lims, lims, '--', color=self.colors.grey, alpha=.75, zorder=0)
                
                #self.fig.text(x=.5, y=0.0, s='PC score (test)', weight='bold', va="center", ha="center")

        def plot_scores(pc_n, pc):
                
            data = df_loadings.xs(key=pc, level='Principal Component')
              
            sign_correction = np.sign( df_reliability.loc[pc][reliability_measure] )
            data_retest = (data['exp']=='retest')
            data.loc[ data_retest , 'Loadings' ] = data.loc[ data_retest , 'Loadings' ].values*sign_correction # CORRECTING SIGN OF RETEST VALUES TO HAVE POSITIVE CORRELATION
            #print(data)
            ax = self.fig.add_subplot(gs[1, pc_n]) if self.join_plots else self.fig.add_subplot(gs[0, 0])
            sns.barplot(data=data, x="variable", y="Loadings", hue="exp", palette=self.colors.palette, ax=ax)
            ax.axhline(y=0, alpha=.3, color=[.5, .5, .5])
            ax.set_ylim([ymin, ymax])
            ax.get_legend().remove() if pc_n > 0 and self.join_plots else None
            ax.set_xlabel('')
            ax.set_ylabel(f'Loadings') if pc_n==0 and self.join_plots else ax.set_ylabel(f'Loadings') if not self.join_plots else ax.set_ylabel('')

        def plot_explained_variance(df_name, method):

            ### Third row: variance explained ####

            df_variance_test = self.fetch.df_variance_explained(exp=0, df_name=df_name, method=method)
            df_variance_test['exp'] = 0

            df_variance_retest = self.fetch.df_variance_explained(exp=1, df_name=df_name, method=method)
            df_variance_retest['exp'] = 1

            # Normalise to 100 %
            df_variance_test['Explained variance'] /= df_variance_test['Explained variance'].sum()
            df_variance_retest['Explained variance'] /= df_variance_retest['Explained variance'].sum()

            df_variance = pd.concat([df_variance_test, df_variance_retest])
            df_variance['Explained variance'] *= 100
            
            #Add zero point just for plotting purposes on lineplot
            data_zero = [['', 0, 0], ['', 0, 1]]
            df_zero = pd.DataFrame(data=data_zero, columns=df_variance.columns)
            df_variance = pd.concat([df_zero, df_variance])
            
            df_variance['exp'] = df_variance['exp'].map(self.exp_dict)

            ax = self.fig.add_subplot(gs[2, :]) if self.join_plots else self.fig.add_subplot(gs[0, 0]) # Span all columns
            
            sns.barplot(data=df_variance, x='Principal Component', y='Explained variance', hue='exp', palette=self.colors.palette, alpha=.7, ax=ax)
            
            data = df_variance.set_index(['exp', 'Principal Component']).groupby(['exp']).cumsum()
            sns.lineplot(data=data, x='Principal Component', y='Explained variance', hue='exp', palette=self.colors.palette, alpha=1, linewidth=5, ax=ax)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Variance Explained (%)')
            ax.legend()
            ax.set_xlim([0, None])
            ax.set_ylim([0, 100.05])

        ####### Plot

        df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_scores', df_name_pca=df_name)

        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure

        # Second row of subplots: PC loadings

        df_loadings_test = self.fetch.df_loadings(exp=0, df_name=df_name, method=method)
        df_loadings_test['exp'] = 0

        df_loadings_retest = self.fetch.df_loadings(exp=1, df_name=df_name, method=method)
        df_loadings_retest['exp'] = 1

        df_loadings = pd.concat([df_loadings_test, df_loadings_retest])
        
        df_loadings['exp'] = df_loadings['exp'].map(self.exp_dict)
                
        ymin = df_loadings.Loadings.min()
        ymax = df_loadings.Loadings.max()
        
        #Adjust plot limits for symmetry
        
        ymin, ymax = self.adjust_limits(ymin, ymax)

        if self.join_plots:
                
            self.fig = plt.figure(figsize=(12, 12))  # Adjusted height to fit additional subplots
            gs = self.fig.add_gridspec(3, 4)

            [plot_loadings(pc_n, pc) for pc_n, pc in enumerate( df_scores.columns )]
                
            # Plot

            df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_scores', df_name_pca=df_name)
            reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
            
            [plot_scores(pc_n, pc) for pc_n, pc in enumerate( df_scores.columns )]
            
            plot_explained_variance(df_name, method)

            plt.tight_layout()
            
            #Save plot
            title = f'Fig2 PCA {df_name} - {self.reliability_measure}'
            self.save.figure(title)

        else:

            for pc_n, pc in enumerate( df_scores.columns ):

                self.fig = plt.figure(figsize=(6, 6))  # Adjusted height to fit additional subplots
                gs = self.fig.add_gridspec(1, 1)

                plot_loadings(pc_n, pc)

                #Save plot
                title = f'Fig2 PCA loadings {pc} {df_name} - {self.reliability_measure}'
                self.save.figure(title)

            # Plot

            df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_scores', df_name_pca=df_name)
            reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
            
            for pc_n, pc in enumerate( df_scores.columns ):

                self.fig = plt.figure(figsize=(6, 6))  # Adjusted height to fit additional subplots
                gs = self.fig.add_gridspec(1, 1)

                plot_scores(pc_n, pc)

                #Save plot
                title = f'Fig2 PCA scores {pc} {df_name} - {self.reliability_measure}'
                self.save.figure(title)

            self.fig = plt.figure(figsize=(6, 6))  # Adjusted height to fit additional subplots
            gs = self.fig.add_gridspec(1, 1)

            plot_explained_variance(df_name, method)
           
            #Save plot
            title = f'Fig2 PCA explained variance {df_name} - {self.reliability_measure}'
            self.save.figure(title)



    def plot_sessions(self):
        
        #Accuracy
        self.df_lc = self.fetch.df_accuracy_sessions()
                        
        self.df_lc.rename(columns=self.learning_contexts_dict, inplace=True) #replace with explicit naming of learning contexts
        self.df_lc.rename(index=self.exp_dict, level='exp', inplace=True)
        self.df_lc = self.df_lc.stack('learning context').to_frame().rename(columns={0: "Accuracy"}).reset_index(['exp', 'session'])
        
        
        ####### Plot
        
        self.fig = plt.figure(figsize=(8, 11))
        
        gs = gridspec.GridSpec(2, 1, figure=self.fig, hspace=0.2)
        
        self.gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], hspace=.4)
        self.gs_base = gs[1].subgridspec(2, 4, hspace=.4)
        
        #################### Accuracy ###################################
        
        df_reliability_A = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy_sessions', experiment=0)
        df_reliability_A['exp'] = 'test'
        df_reliability_B = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy_sessions', experiment=1)
        df_reliability_B['exp'] = 'retest'
        self.df_reliability = pd.concat([df_reliability_A, df_reliability_B])
        
        self.panel_B('test')     
        self.panel_B('retest')
        
        df_reliability_A = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy_sessions', session=0)
        df_reliability_A['session'] = 0
        df_reliability_B = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy_sessions', session=1)
        df_reliability_B['session'] = 1
        self.df_reliability = pd.concat([df_reliability_A, df_reliability_B])
        
        self.panel_B('Session 1')
        self.panel_B('Session 2')
        
        self.figure_text(sessions=True)
        self.panel_indexing(sessions=True)

        #Save or plot
        title = f'Fig2 sessions - {self.reliability_measure}'
        self.save.figure(title)

    def plot_RT(self):
                
        self.df_lc = self.fetch.df_RT()

        self.df_lc.rename(columns=self.learning_contexts_dict, inplace=True) #replace with explicit naming of learning contexts
        self.df_lc.rename(index=self.exp_dict, level='exp', inplace=True)
        self.df_lc = self.df_lc.stack('learning context').to_frame().rename(columns={0: "Reaction time"}).reset_index('exp')

        self.df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_RT')
                
        ####### Plot
                
        self.fig = plt.figure(figsize=(11, 7))
        
        gs = gridspec.GridSpec(1, 1, figure=self.fig)
        
        self.gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], hspace=0.3)
            
        #################### Accuracy ###################################
    
        self.panel_A()     
        self.panel_B() #Correlations
            
        self.fig.text(x=.5, y=0.05, s=f'Test reaction time (s)', weight='bold', va="bottom", ha="center")
        self.panel_indexing(n=2)

        #Save
        title = f'Fig2 RT - {self.reliability_measure}'
        self.save.figure(title)    

    def plot(self, method):
        
        self.method = method

        #Accuracy
        self.df_lc = self.fetch.df_accuracy()
                        
        self.df_lc.rename(columns=self.learning_contexts_dict, inplace=True) #replace with explicit naming of learning contexts
        
        self.df_lc.rename(index=self.exp_dict, level='exp', inplace=True)
        
        self.df_lc = self.df_lc.stack('learning context').to_frame().rename(columns={0: "Accuracy"}).reset_index('exp')
        
        self.df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy')

        #Parameters
        
        self.df_parameters = self.fetch.df_parameters(method=self.method, pool_exp=False)
        self.df_parameters.rename(columns=self.labels_to_replace, level='parameter', inplace=True) #no italics for greek letters
                
        self.parameters = list(self.df_parameters.columns)
        self.df_parameters = self.df_parameters.stack('parameter').to_frame().rename(columns={0: "Fit"})
        self.df_parameters.rename(index=self.exp_dict, level='exp', inplace=True)
                
        ####### Plot
                
        self.fig = plt.figure(figsize=(11, 11))
        
        gs = gridspec.GridSpec(2, 1, figure=self.fig, hspace=0.3)
        
        self.gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0])
        self.gs_base = gs[1].subgridspec(2, 4)
            
        #################### Accuracy ###################################
    
        self.panel_A()     
        self.panel_B() #Correlations
        
        #################### Parameters ###################################
    
        self.panel_C()
        self.panel_D() #Correlations
    
        self.figure_text()
        self.panel_indexing()

        #Save
        title = f'Fig2 - {self.reliability_measure} - RL parameter fit by {self.method}'
        self.save.figure(title)

    def plot_rebuttal_fig14(self, folder, method_list):
                                 
        ####### Plot
                
        self.fig = plt.figure(figsize=(9, 9))
        
        gs = gridspec.GridSpec(2, 1, figure=self.fig, hspace=0.35)
        
        self.gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0])
        self.gs_base = gs[1].subgridspec(2, 4)
            
        #################### Accuracy ###################################

        self.method = method_list[0]
        super().__init__(self.method) #fetch from folder

        #Parameters
        
        self.df_parameters = self.fetch.df_parameters(method=self.method, pool_exp=False)
        self.df_parameters.rename(columns=self.labels_to_replace, level='parameter', inplace=True) #no italics for greek letters
                
        self.parameters = list(self.df_parameters.columns)
        self.df_parameters = self.df_parameters.stack('parameter').to_frame().rename(columns={0: "Fit"})
        self.df_parameters.rename(index=self.exp_dict, level='exp', inplace=True)
    
        self.panel_C(default=False)     
        self.panel_D(default=False) #Correlations
        
        #################### Parameters ###################################

        self.method = method_list[1]
        super().__init__(self.method) #fetch from folder
        
        #Parameters
        
        self.df_parameters = self.fetch.df_parameters(method=self.method, pool_exp=False)
        self.df_parameters.rename(columns=self.labels_to_replace, level='parameter', inplace=True) #no italics for greek letters
                
        self.parameters = list(self.df_parameters.columns)
        self.df_parameters = self.df_parameters.stack('parameter').to_frame().rename(columns={0: "Fit"})
        self.df_parameters.rename(index=self.exp_dict, level='exp', inplace=True)
    
        self.panel_C()
        self.panel_D() #Correlations
    
        self.panel_indexing()

        self.fig.text(x=.5, y=0.06, s='parameter estimate (test)', weight='bold', va="bottom", ha="center")
        self.fig.text(x=.5, y=0.51, s='parameter estimate (test)', weight='bold', va="center", ha="center")

        super().__init__(folder) #save in joint folder

        #Save
        title = f'Fig2 - {self.reliability_measure} - RL parameter fit by {method_list[0]} (A-B) {method_list[1]} (C-D)'
        self.save.figure(title)

    def plot_main_contrasts(self):

        #Accuracy
        self.df_lc = self.fetch.df_accuracy_main_contrasts()
                
        #self.df_lc.rename(columns=self.learning_contexts_dict, inplace=True) #replace with explicit naming of learning contexts
        self.df_lc.rename(index=self.exp_dict, level='exp', inplace=True)
        #self.df_lc = self.df_lc.stack('learning context').to_frame().rename(columns={0: "Accuracy"}).reset_index('exp')
        
        self.df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_accuracy_main_contrasts')
                        
        ####### Plot
                
        self.fig = plt.figure(figsize=(11, 6))
        
        gs = gridspec.GridSpec(1, 1, figure=self.fig, hspace=0.3)
        self.gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
            
        #################### Accuracy ###################################
      
        self.panel_B2(ylabel='Accuracy contrast (retest) accuracy (%)') #Correlations
        
        self.fig.text(x=.5, y=0.05, s='Accuracy contrast (test) accuracy (%)', weight='bold', va="center", ha="center")
        
        x_coordinate = [0.07, 0.50] 
        y = .89
        [plt.figtext(x, y, string.ascii_uppercase[n]+'.', weight='bold') for n, x in enumerate(x_coordinate)]        

        #Save
        title = f'Fig2 - {self.reliability_measure} - main contrasts'
        self.save.figure(title)

    def plot_main_contrast_reaction_time(self):

        #Reaction time
        self.df_lc = self.fetch.df_reaction_time_main_contrasts()
                
        self.df_lc.rename(index=self.exp_dict, level='exp', inplace=True)
        
        self.df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_reaction_time_main_contrasts')
                        
        ####### Plot
                
        self.fig = plt.figure(figsize=(11, 6))
        
        gs = gridspec.GridSpec(1, 1, figure=self.fig, hspace=0.3)
        self.gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
            
        #################### Accuracy ###################################
      
        self.panel_B2(ylabel='Reaction time contrast (s) (retest)') #Correlations
        
        self.fig.text(x=.5, y=0.05, s='Reaction time contrast (s) (test)', weight='bold', va="center", ha="center")
        
        x_coordinate = [0.07, 0.50] 
        y = .89
        [plt.figtext(x, y, string.ascii_uppercase[n]+'.', weight='bold') for n, x in enumerate(x_coordinate)]        

        #Save
        title = f'Fig2 - {self.reliability_measure} - main contrasts time'
        self.save.figure(title)
                
    def ax_limits(self, i, variable, ax):
        
        if 'alpha' in variable:
            
            max_axis = 1
            ax.set(xlim=(-.05, 1), ylim=(-.05, 1), yticks=self.tick, yticklabels=self.tick) if i == 1 else ax.set(xlim=(-.05, 1), ylim=(-.05, 1), yticks=self.tick, yticklabels=['', '', '']) if i > 1 else None
              
        elif 'beta' in variable:
                
            max_axis = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.set_xlim(-0.05, max_axis)
            ax.set_ylim(-0.05, max_axis)
            
        # Draw a line of x=y 
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        
        ax.plot(lims, lims, '--', color=self.colors.grey, alpha=.5, linewidth=1)
    
    def figure_text(self, sessions=False):
                
        if not sessions:
        
            self.fig.text(x=.5, y=0.06, s='parameter estimate \n (test)', weight='bold', va="bottom", ha="center")
            self.fig.text(x=.5, y=0.51, s='test accuracy (%)', weight='bold', va="center", ha="center")
                    
        else:

            self.fig.text(x=.5, y=0.70, s='Session 1 (test) accuracy (%)', weight='bold', va="center", ha="center")
            self.fig.text(x=.5, y=0.49, s='Session 1 (retest) accuracy (%)', weight='bold', va="center", ha="center")
            #self.fig.text(x=.5, y=0.47, s='Between-experiments', weight='bold', va="center", ha="center")
            self.fig.text(x=.5, y=0.28, s='Session 1 (test) accuracy (%)', weight='bold', va="center", ha="center")
            self.fig.text(x=.5, y=0.08, s='Session 2 (test) accuracy (%)', weight='bold', va="center", ha="center")
            
    def panel_indexing(self, n=4, sessions=False):
    
        x = 0.06 if not sessions else 0.02
        y_coordinate = [.89, .70, .46, .27] if n == 4 else [.89, .46] if n == 2 else None
    
        [plt.figtext(x, y, string.ascii_uppercase[n]+'.', weight='bold', fontsize=13) for n, y in enumerate(y_coordinate)]     

    def text_reliability(self, reliability_measure, variable, df_reliability, ax):

        r = df_reliability.loc[variable][reliability_measure]
        p_star = df_reliability.loc[variable]['p*']
        
        x_r = .52*ax.get_xlim()[1]
        y_r = .15*ax.get_ylim()[1]
        
        ax.text(x=x_r, y=y_r, s=f"{reliability_measure} = "+"{:.2f}".format(r)+p_star, bbox ={'facecolor':'white',
                                                                                           'edgecolor': 'white',
                                                                                           'alpha':0.6,
                                                                                           'pad':6}) 
        
        return

    def panel_A(self):
        
        variable = self.df_lc.columns[1]
        
        tick = [0, 25, 50, 75, 100] if variable == 'Accuracy' else [0.5, 1, 1.5, 2, 2.5]
    
        for learning_context_number, (learning_context, learning_context_verbose) in enumerate(self.learning_contexts_dict.items()):
    
            ax1 = self.fig.add_subplot(self.gs_top[0, learning_context_number]) #fig.add_subplot(spec[0, lcn])
            
            data = self.df_lc.xs(learning_context_verbose, level='learning context')
            
            ### Kernel shade
        
            violinplot = sns.violinplot(x="exp", y=variable, data=data, order=self.hue_order_exp, inner=None, legend=False, ax = ax1, color=self.colors.learning_contexts[learning_context]) #split=True
            [violin.set_alpha(.4) for i, violin in enumerate(violinplot.collections)]
    
            ### Dots
        
            dots = sns.stripplot(x="exp", y=variable, data=data, order=self.hue_order_exp, alpha=.5, ax = ax1, palette=self.colors.palette)

            #for i, dot in enumerate([p for p in dots.patches if not p.get_label()]): 
            #    color = dot.get_facecolor()
                #box.set_edgecolor(color)
            #    dot.set_facecolor((0, 0, 0, 0))
    
            ### Box
                  
            boxplot = sns.boxplot(x="exp", y=variable, data=data, order=self.hue_order_exp, showfliers = False, whis=0, linewidth=3, ax = ax1, color=self.colors.learning_contexts[learning_context], boxprops={'zorder': 3})
            
            for i,box in enumerate([p for p in boxplot.patches if not p.get_label()]): 
                color = box.get_facecolor()
                box.set_edgecolor(color)
                box.set_facecolor((0, 0, 0, 0))
                # iterate over whiskers and median lines
                for j in range(5*i,5*(i+1)):
                    boxplot.lines[j].set_color(color)
                    boxplot.lines[j].set_zorder(3)
            
            ### Mean
            
            df_mean = data.groupby(by='exp').mean().reset_index()
            
            sns.stripplot(x="exp", y=variable, data=df_mean, hue_order=self.hue_order_exp, color=self.colors.learning_contexts[learning_context],
                                marker='o', size=self.point_size, jitter=False, dodge=True, edgecolor='k', linewidth=1, label='mean', ax = ax1)
        
            ### Customise plot
        
            ax1.get_legend().remove()                   

            ax1.spines['bottom'].set_visible(False)
            ax1.xaxis.set_ticks_position('none')
            
            ylabel = ''
            
            if learning_context_number == 0:
            
                if variable == 'Accuracy':
                    ylabel = 'accuracy (%)'
                else:
                    ylabel = f'{variable} (s)'
                
            ax1.set_ylabel(ylabel)
            
            if variable == 'Accuracy':
                ax1.set_ylim([0, 100])
                ax1.axhline(y=50, color='k', linestyle='-', linewidth=0.7) #line at chance level
            
            else:
                ax1.set_ylim([0, data[variable].max()*1.02 ])
                
            ax1.set_xlabel('')
            ax1.set_title(learning_context_verbose, fontsize=12)
            
            ax1.set_yticks(tick)
            ax1.set_yticklabels(tick) if learning_context_number == 0 else ax1.set_yticklabels([])


    def panel_B(self, exp=None):
                
        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
        
        for learning_context_number, (learning_context, learning_context_verbose) in enumerate(self.learning_contexts_dict.items()):
            
            if not exp:
                
                variable = self.df_lc.columns[1]
                
                ax = self.fig.add_subplot(self.gs_top[1, learning_context_number])
                
                data = self.df_lc.pivot_table(values=variable, columns='exp', index=['id', 'learning context']).xs(learning_context_verbose, level='learning context')
                
                x, y = 'test', 'retest'
                
            else:
                
                variable = self.df_lc.columns[2]
                
                if exp == 'test':
                    ax = self.fig.add_subplot(self.gs_top[0, learning_context_number])                    
                elif exp == 'retest':
                    ax = self.fig.add_subplot(self.gs_top[1, learning_context_number])
                elif exp == 'Session 1':
                    ax = self.fig.add_subplot(self.gs_base[0, learning_context_number])
                elif exp == 'Session 2':
                    ax = self.fig.add_subplot(self.gs_base[1, learning_context_number])
                
                if exp in ['test', 'retest']:    
                    data = self.df_lc[self.df_lc.exp==exp].pivot_table(values=variable, columns='session', index=['id', 'learning context']).xs(learning_context_verbose, level='learning context')
                    data.rename(columns={0: 'session 1', 1: 'session 2'}, inplace=True)
                    x, y = 'session 1', 'session 2'
                
                elif exp in ['Session 1', 'Session 2']:

                    dict_session = dict({'Session 1': 0, 'Session 2': 1})
                    session = dict_session[exp]
                    data = self.df_lc[self.df_lc.session==session].pivot_table(values=variable, columns='exp', index=['id', 'learning context']).xs(learning_context_verbose, level='learning context')
                    x, y = 'test', 'retest'
                
                
            g = sns.regplot(x=x, y=y,
                           data=data,
                           color=self.colors.learning_contexts[learning_context],
                           scatter_kws={'s': self.point_size},
                           line_kws={'color': 'k', 'lw': 2},
                           ax = ax)

            g.set_title(learning_context_verbose, fontsize=12) if exp=='test' else None
        
            g.set_xlabel('')
            g.set_ylabel('')
                        
            if variable == "Accuracy":
            
                tick = [0, 50, 100]
            
                if learning_context_number==0:
            
                    g.set_ylabel('re-test accuracy (%)') if not exp else g.set_ylabel(f'Session 2 ({exp}) \n accuracy (%)') if exp in ['test', 'retest'] else g.set_ylabel(f'{exp} (retest) \n accuracy (%)') if exp in ['Session 1', 'Session 2'] else None
                    g.set(xlim=(-.05, 102), ylim=(-.05, 102), xticks=tick, xticklabels=tick, yticks=tick, yticklabels=tick)
                
                else:
            
                    g.set(xlim=(-.05, 102), ylim=(-.05, 102), xticks=tick, xticklabels=tick, yticks=tick, yticklabels=[])
            else:
                
                if learning_context_number==0:
                    g.set_ylabel(f'Retest reaction time (s)')
                    g.set(xlim=(0, self.df_lc[variable].max()*1.05), ylim=(0, self.df_lc[variable].max()*1.02))
                else:
                    g.set(xlim=(0, self.df_lc[variable].max()*1.05), ylim=(0, self.df_lc[variable].max()*1.02), yticklabels=[])
                    
                # Loop to write R and draw a line of x=y
            if exp is None:
                df_reliability = self.df_reliability
            elif exp in ['test', 'retest']:
                df_reliability = self.df_reliability[self.df_reliability.exp==exp] 
            elif exp in ['Session 1', 'Session 2']:
                df_reliability =  self.df_reliability[self.df_reliability.session==session]
            
            self.text_reliability(reliability_measure, learning_context, df_reliability, g)
    
            # Draw a line of x=y
            x0, x1 = g.get_xlim()
            y0, y1 = g.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            g.plot(lims, lims, '--', color=self.colors.grey, alpha=.75, zorder=0)

    def panel_B2(self, ylabel):
                
        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
        
        variables = self.df_lc.index.get_level_values('variable').unique()
        lims_ = self.df_lc['contrast'].max()*1.05

        df_reliability = self.df_reliability
                    
        for variable_n, variable in enumerate( variables ):
        
            data = self.df_lc.xs(variable).unstack('exp')['contrast']

            ax = self.fig.add_subplot(self.gs_top[0, variable_n])    
            x, y = 'test', 'retest'
                
            g = sns.regplot(x=x, y=y,
                           data=data,
                           color=self.colors.grey,
                           scatter_kws={'s': self.point_size},
                           line_kws={'color': 'k', 'lw': 2},
                           ax = ax)

            g.set_title(variable, fontsize=12)
        
            g.set_xlabel('')
            g.set_ylabel(ylabel) if variable_n==0 else g.set_ylabel('')
                        
            g.set(xlim=(-lims_, lims_), ylim=(-lims_, lims_))
            
            plt.axhline(0, color=self.colors.grey, alpha=.3, lw=1)
            plt.axvline(0, color=self.colors.grey, alpha=.3, lw=1)
            
            # Loop to write R and draw a line of x=y            
            self.text_reliability(reliability_measure, variable, df_reliability, g)
    
            # Draw a line of x=y
            x0, x1 = g.get_xlim()
            y0, y1 = g.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            g.plot(lims, lims, '--', color=self.colors.grey, alpha=.75, zorder=0)          

    def panel_C(self, default=True):
        
        #df_par = df_par.apply(lambda x : x/x.max() ) #normalise
    
        tick = [0, .5, 1] #for alphas 
                
        for parameter_n, parameter in enumerate(self.parameters):
            
            subplot = self.gs_base[0, parameter_n] if default else self.gs_top[0, parameter_n] 
            ax3 = self.fig.add_subplot(subplot) #fig.add_subplot(spec[2, p])
            
            data = self.df_parameters.reset_index('exp').xs(parameter, level='parameter')

            ### Kernel shade
            
            violinplot_par = sns.violinplot(x="exp", y="Fit", data=data, order=self.hue_order_exp, palette=self.colors.palette, inner=None, legend=False, ax = ax3) #split=True
            [violin.set_alpha(.4) for i, violin in enumerate(violinplot_par.collections)]
        
            ### Dots

            dots = sns.stripplot(x="exp", y="Fit", data=data, order=self.hue_order_exp, alpha=.5, ax = ax3, palette=self.colors.palette)               

            ### Box
                  
            boxplot_par = sns.boxplot(x="exp", y="Fit", data=data, order=self.hue_order_exp, showfliers = False, whis=0, linewidth=3, ax = ax3, palette=self.colors.palette)
        
            for i,box_par in enumerate([p for p in boxplot_par.patches if not p.get_label()]): 
                color = box_par.get_facecolor()
                box_par.set_edgecolor(color)
                box_par.set_facecolor((0, 0, 0, 0))
                # iterate over whiskers and median lines
                for j in range(5*i,5*(i+1)):
                    boxplot_par.lines[j].set_color(color)
                    
            ### Mean
                    
            df_mean = data.groupby('exp').mean().reset_index()
            df_mean['parameter'] = parameter
            
            stripplot_par = sns.stripplot(x="exp", y="Fit", data=df_mean, order=self.hue_order_exp,
                                marker='o', size=10, jitter=False, color='w', edgecolor='k', linewidth=1, label='mean', ax = ax3) #alpha=.8, dodge=True,
    
            ax3.set_xlabel('')
            ax3.set_ylabel('parameter estimate') if parameter_n == 0 else ax3.set_ylabel('')
            ax3.set_ylim([0, None]) if parameter_n == 0 else ax3.set_ylim([0, 1])
            ax3.spines['bottom'].set_visible(False)
                        
            ax3.set_title(parameter, y=1.07)
            ax3.set_yticks(tick) if parameter_n in [1, 2, 3] else None
            ax3.set_yticklabels([]) if parameter_n in [2, 3] else ax3.set_yticklabels(tick) if parameter_n == 1 else None
            
            ax3.get_legend().remove()

    def panel_D(self, default=True):
        
        self.df_parameters = self.df_parameters.pivot_table(values='Fit', columns='exp', index=['id', 'parameter'])
        
        df_reliability = self.fetch.df_reliability(self.reliability_measure, 'df_parameters', pool_exp=False, method=self.method)
        df_reliability.rename(index=self.labels_to_replace, inplace=True) #no italics for greek letters
        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
        
        self.tick = [0, .5, 1]
        
        for parameter_n, parameter in enumerate( self.parameters ):
            
            subplot = self.gs_base[1, parameter_n] if default else self.gs_top[1, parameter_n]
            ax4 = self.fig.add_subplot(subplot)
            
            data = self.df_parameters.xs(parameter, level='parameter')
                
            g3 = sns.regplot(x="test", y="retest",
                           data=data,
                           color=self.colors.grey,
                           scatter_kws={'s': self.point_size},
                           line_kws={'color': 'k', 'lw': 2},
                           ax = ax4)
       
            g3.set_xlabel('')
            g3.set_ylabel('parameter estimate \n (re-test)') if parameter_n==0 else g3.set_ylabel('')
     
            self.ax_limits(parameter_n, parameter, ax4)

            # Loop to write R and draw a line of x=y
            self.text_reliability(reliability_measure, parameter, df_reliability, g3)
            
            # Draw a line of x=y 
            x0, x1 = ax4.get_xlim()
            y0, y1 = ax4.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            g3.plot(lims, lims, '--', color=self.colors.grey, alpha=.75, zorder=0)


    def adjust_limits(self, data_min, data_max):
    
        if (data_min < 0)*(data_max>0):

            data_min *= 1.05
            data_max *= 1.05

            data_max = max(abs(data_min), data_max)  
            data_min = -max(abs(data_min), data_max)
            
        elif (data_min > 0)*(data_max>0):
        
            data_min *= 0.95
            data_max *= 1.05

        elif (data_min < 0)*(data_max<0):
        
            data_min *= 1.05
            data_max = 0
            
            
        return data_min, data_max

class Figure3(Plot):
    
    def __init__(self, reliability_measure, folder):
                
        super().__init__(folder)
        
        self.reliability_measure = reliability_measure
        
        from matplotlib.ticker import FormatStrFormatter

        self.FormatStrFormatter = FormatStrFormatter
        
        df_propensity = self.fetch.df_propensity()
        df_clinics = self.fetch.df_clinics()

        self.df_list = dict({'df_propensity': df_propensity,
                             'df_clinics': df_clinics})
        
    def plot(self):
        
        #fig = plt.figure(figsize=(10,10), constrained_layout=True)
        #gs_top = plt.GridSpec(2, 4, bottom=.535, top=0.95, hspace=.2, left=0.075)
        #gs_base = plt.GridSpec(2, 4,  bottom=.05, top=0.465, hspace=.2, left=0.075)

        # gridspec inside gridspec
        fig = plt.figure(figsize=(11, 11))
        
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
        
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0])
        gs_base = gs[1].subgridspec(2, 4)
        
        for df_number, df_item in enumerate(self.df_list.items()):
            
            df_name = df_item[0]
            df = df_item[1]
                                
            df_reliability = self.fetch.df_reliability(self.reliability_measure, df_name)
            reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
            
            variables = df.columns.tolist() #get variables' names
            
            df = df.reset_index()
            df['exp'] = df['exp'].map(self.exp_dict) #convert 0 and 1 to text (test and retest)
            
            for variable_n, variable in enumerate(variables):
                                
                if variable=='Alc':
                    max_axis = 40
                    
                elif variable=='Nic':
                    max_axis = 10
                        
                elif variable=='Anx' or variable=='Dep':
                    max_axis = 21
                        
                elif variable=='BASr':
                    max_axis = 20
                        
                elif variable=='BASd' or variable=='BASf':
                    max_axis = 16
                        
                elif variable=='BIS':
                    max_axis=28
                                
                ######
                                
                ax1 = fig.add_subplot(gs_top[0, variable_n]) if df_number == 0 else fig.add_subplot(gs_base[0, variable_n])              
                                
                ### Kernel shade
                
                violinplot_par = sns.violinplot(x="exp", y=variable, data=df, order=self.hue_order_exp, palette=self.colors.palette, inner=None, legend=False, ax = ax1) #split=True
                [violin.set_alpha(.4) for i, violin in enumerate(violinplot_par.collections)]
            
                ### Dots
                    
                dots = sns.stripplot(x="exp", y=variable, data=df, hue='exp', hue_order=self.hue_order_exp, alpha=.5, ax = ax1, palette=self.colors.palette, legend=False)               
                #ax1.get_legend().remove()
                
                ### Box
                      
                boxplot = sns.boxplot(x="exp", y=variable, data=df, order=self.hue_order_exp, showfliers = False, whis=0, linewidth=3, ax = ax1, palette=self.colors.palette)
            
                for i,box in enumerate([p for p in boxplot.patches if not p.get_label()]): 
                    color = box.get_facecolor()
                    box.set_edgecolor(color)
                    box.set_facecolor((0, 0, 0, 0))
                    # iterate over whiskers and median lines
                    [boxplot.lines[j].set_color(color) for j in range(5*i,5*(i+1))]
                    
                ### Mean

                df_mean = df.groupby('exp')[variable].mean().reset_index()
                
                sns.stripplot(x="exp", y=variable, data=df_mean, order=self.hue_order_exp, marker='o', size=self.point_size, jitter=False, color='w', edgecolor='k', linewidth=1, ax = ax1) #alpha=.8, dodge=True,
                
                ### Refine
                ax1.set_title(variable)
                #ax1.get_legend().remove()
                ax1.set_yticks(np.linspace(0, max_axis, 3))
                ax1.set_yticklabels(np.linspace(0, max_axis, 3))
                ax1.yaxis.set_major_formatter(self.FormatStrFormatter('%.0f')) 
                #ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax1.set_ylim(-1/55*max_axis, max_axis+1)
                ax1.spines['bottom'].set_visible(False)
                ax1.set_xlabel('')
                ax1.set_ylabel('Score') if variable_n == 0 else ax1.set_ylabel('')
                
                ######

                ax2 = fig.add_subplot(gs_top[1, variable_n]) if df_number == 0 else fig.add_subplot(gs_base[1, variable_n])
    
                sns.regplot(x="test", y="retest",
                               data=df.set_index(['exp', 'id'])[variable].unstack('exp'),
                               color=self.colors.grey,
                               scatter_kws={'alpha':0.2}, #"s": 'size'},
                               line_kws={'color': 'k', 'lw': 2},
                               ax=ax2)
                                                                    
                ax2.set_xlim(0-1/55*max_axis, max_axis+1)
                ax2.set_ylim(0-1/55*max_axis, max_axis+1)
                            
                ax2.set_yticks(np.linspace(0, max_axis, 3))
                ax2.set_yticklabels(np.linspace(0, max_axis, 3))
                ax2.yaxis.set_major_formatter(self.FormatStrFormatter('%.0f'))                                            
                
                ax2.set_xlabel('')
                ax2.set_ylabel('Score (retest)') if variable_n==0 else ax2.set_ylabel('')
                
                # Draw a line of x=y 
                x0, x1 = ax2.get_xlim()
                y0, y1 = ax2.get_ylim()
                lims = [min(x0, y0), max(x1, y1)]
                
                ax2.plot(lims, lims, '--', color=self.colors.grey, alpha=.5, linewidth=1)
                            
                # Loop to write R and draw a line of x=y
                
                r = df_reliability.loc[variable][reliability_measure]
                p_star = df_reliability.loc[variable]['p*']
                
                x_r = .52*ax2.get_xlim()[1]
                y_r = .15*ax2.get_ylim()[1]
                                
                ax2.text(x=x_r, y=y_r, s=f"{reliability_measure} ="+"{:.2f}".format(r)+p_star)
                
                #ax2.subplots_adjust(bottom=0.2) if df_number == 0 else None
                ax2.margins(0.25) if df_number == 0 else None
                
        fig.text(x=.5, y=0.05, s="Score (test)", weight='bold', va="bottom", ha="center")
        fig.text(x=.5, y=0.50, s="Score (test)", weight='bold', va="center", ha="center")
        
        plt.figtext(0.07, .89, string.ascii_uppercase[0]+'.', weight='bold')
        plt.figtext(0.07, .69, string.ascii_uppercase[1]+'.', weight='bold')
        plt.figtext(0.07, .46, string.ascii_uppercase[2]+'.', weight='bold')
        plt.figtext(0.07, .27, string.ascii_uppercase[3]+'.', weight='bold')
                
        title = f'Fig3 - {self.reliability_measure}'
        self.save.figure(title)            
        
################### CROSS-CORRELATION ANALYSIS #############################

class Figure4(Plot):
    
    def __init__(self, reliability_measure, method, first_df='accuracy'):
        
        super().__init__(method) #saving folder
                
        self.method = method
        self.reliability_measure = reliability_measure
        
        self.reliability_measure_label = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
                        
        cross_correlation_dict = self.fetch.df_cross_correlation_dict(0, method=method, first_df=first_df)
                
        cross_correlation_dict = self.no_italics_greek(cross_correlation_dict)
        
        self.variables_order = cross_correlation_dict['R'].columns.to_list()

        cross_correlation_dict = self.fetch.df_cross_correlation_dict('combined', method=method, first_df=first_df)
        cross_correlation_dict = self.no_italics_greek(cross_correlation_dict)
        df_combined_dict = dict({'R': self.lower_triangular(cross_correlation_dict['R']),
                             'p': self.lower_triangular(cross_correlation_dict['p']),
                             'p_NotCorrected': self.lower_triangular(cross_correlation_dict['p_NotCorrected'])})
        
        self.first_df = first_df

        #if first_df == 'accuracy':

        df_test_dict = dict({'R': self.lower_triangular(cross_correlation_dict['R']),
                             'p': self.lower_triangular(cross_correlation_dict['p']),
                             'p_NotCorrected': self.lower_triangular(cross_correlation_dict['p_NotCorrected'])})

        cross_correlation_dict = self.fetch.df_cross_correlation_dict(1, method=method, first_df=first_df)
        cross_correlation_dict = self.no_italics_greek(cross_correlation_dict)
        df_retest_dict = dict({'R': self.lower_triangular(cross_correlation_dict['R']),
                             'p': self.lower_triangular(cross_correlation_dict['p']),
                             'p_NotCorrected': self.lower_triangular(cross_correlation_dict['p_NotCorrected'])})

        cross_correlation_dict = self.fetch.df_cross_correlation_dict('difference', method=method, first_df=first_df)
        cross_correlation_dict = self.no_italics_greek(cross_correlation_dict)
        df_difference_dict = dict({'R': self.lower_triangular(cross_correlation_dict['R']),
                             'p': self.lower_triangular(cross_correlation_dict['p']),
                             'p_NotCorrected': self.lower_triangular(cross_correlation_dict['p_NotCorrected'])})
        
        self.df_dict = dict({'test': df_test_dict, 'retest': df_retest_dict, 'combined': df_combined_dict, 'difference': df_difference_dict})

        self.alpha_pvalue_NotCorrected = .35

        #else:

        #    self.df_dict = dict({'combined': df_combined_dict})

    def no_italics_greek(self, cross_correlation_dict):
    
        cross_correlation_dict['R'].rename(columns=self.labels_to_replace, index=self.labels_to_replace, inplace=True)
        cross_correlation_dict['p'].rename(columns=self.labels_to_replace, index=self.labels_to_replace, inplace=True)
        cross_correlation_dict['p_NotCorrected'].rename(columns=self.labels_to_replace, index=self.labels_to_replace, inplace=True)
        
        return cross_correlation_dict
 
    def class_measure(self, df, color, label, ax):
        
        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
        
        df.plot.scatter(reliability_measure, 'CV', marker='s', color=color, label=label, ax = ax)

        #add annotations
        text_x, text_y = np.zeros(4), np.zeros(4)
        [plt.annotate(df.index[idx_n], xy=xy, xytext=(xtxt, ytxt), textcoords="offset points") for idx_n, (xy, xtxt, ytxt) in enumerate(zip(zip(df[f'{reliability_measure}'], df.CV), text_x, text_y))]
   
    def reliability_vs_CV(self, position):

        self.df_accuracy = self.fetch.df_reliability_vs_CV(self.reliability_measure, f'df_{self.first_df}')
        self.df_propensity = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_propensity')
        self.df_clinics = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_clinics')
        self.df_parameters = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_parameters', method=self.method)
        
        self.df_parameters.rename(self.labels_to_replace, inplace=True)
        
        ax, _ = self.set_grid(self.f, position=position)
               
        ### Accuracy
        self.class_measure(self.df_accuracy, '#cc0000', 'Behavioural measures', ax)
       
        ### Parameters
        self.class_measure(self.df_parameters, '#ffae25', 'Computational measures', ax)
      
        ### Clinical
        self.class_measure(self.df_clinics, '#008080', 'Clinical measures', ax)
      
        ### Propensity
        self.class_measure(self.df_propensity, '#0b5394', 'Propensity measures', ax)
       
        plt.xlabel(f'Test-retest {self.reliability_measure}')
        plt.ylabel('Coefficient of variation')
   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
   
        ax.set_xlim([0, 1])
       
        leg = ax.legend(handletextpad=0.1)
        leg.get_frame().set_linewidth(0.0)       

    def plot(self):
        
        '''
        Function to assemble all the parts of figures
        '''
                    
        ### Figure
        
        self.f = plt.figure(figsize=(10.5, 10.5))
        #f, axs = plt.subplots(1, 3, figsize=(5.5, 5.5))
    
        self.size_scale = 250 #size squares
        self.size_circles_big = 40
        self.size_circles_small = 20
        
        self.color_range = [-1, 1]
        self.palette = sns.diverging_palette(0, 250, n=256)
        self.size_range = [0,1]
        self.marker = 's'

        #Top left            
        self.reliability_vs_CV(position=0)
        
        #Top right
        self.overlay('combined', position=1, colorbar=True)
            
        #Bottom left of Grid
        self.overlay('test', position=2)
    
        #Bottom right of Grid
        self.overlay('retest', position=3, colorbar=True)
            
        plt.figtext(0.07, .90, string.ascii_uppercase[0]+'.', fontsize=13, weight='bold', va="top", ha="left")
        plt.figtext(.5, .90, string.ascii_uppercase[1]+'.', fontsize=13, weight='bold', va="top", ha="center")
        plt.figtext(0.07, .48, string.ascii_uppercase[2]+'.', fontsize=13, weight='bold', va="bottom", ha="center")
        plt.figtext(.5, .48, string.ascii_uppercase[3]+'.', fontsize=13, weight='bold', va="bottom", ha="center")
        
        title = f'Fig4 - {self.reliability_measure} in subplot A - first df is {self.first_df} - RL parameters fit by {self.method}'
        self.save.figure(title)

    def plot_difference(self):
        
        '''
        Function to assemble all the parts of figures
        '''
                    
        ### Figure
        
        self.f = plt.figure(figsize=(10.5, 10.5))
        #f, axs = plt.subplots(1, 3, figsize=(5.5, 5.5))
    
        self.size_scale = 250 #size squares
        self.size_circles_big = 40
        self.size_circles_small = 20
        
        self.color_range = [-1, 1]
        self.palette = sns.diverging_palette(0, 250, n=256)
        self.size_range = [0,1]
        self.marker = 's'
                    
        #Bottom left of Grid
        self.overlay('test', position=0)
    
        #Bottom right of Grid
        self.overlay('retest', position=1, colorbar=True)

        #Top right
        self.overlay('combined', position=2)

        #Top right
        self.overlay('difference', position=3, colorbar=True)

        plt.figtext(0.07, .90, string.ascii_uppercase[0]+'.', weight='bold', va="top", ha="left")
        plt.figtext(.5, .90, string.ascii_uppercase[1]+'.', weight='bold', va="top", ha="center")
        plt.figtext(0.07, .48, string.ascii_uppercase[2]+'.', weight='bold', va="bottom", ha="center")
        plt.figtext(.5, .48, string.ascii_uppercase[3]+'.', weight='bold', va="bottom", ha="center")
        
        title = f'Fig4 - first df is {self.first_df} - RL parameters fit by {self.method} - test retest difference in subplot D'
        self.save.figure(title)

    def overlay(self, experiment, position, label=True, colorbar=False):
        
        labels = ['significant p-value', 'significant p-value (corrected)'] if label else ['_nolegend_', '_nolegend_']
        
        R = self.df_dict[experiment]['R']
        pvalues = self.df_dict[experiment]['p']
        pvalues_NotCorrected = self.df_dict[experiment]['p_NotCorrected']
        
        f, ax = self.heatmap(
            R['x'], R['y'], self.f, colorbar, position, experiment,
            color=R['value'],
            color_range=self.color_range,
            palette=self.palette,
            size=R['value'].abs(),
            size_range=self.size_range,
            marker=self.marker,
            x_order=self.variables_order,
            y_order=self.variables_order[::-1],
            size_scale=self.size_scale)
        
        self.heatmap_pvalues(
            R['x'], R['y'], f, position, ax,
            size_scale=self.size_circles_small,
            alpha=pvalues_NotCorrected['value']*self.alpha_pvalue_NotCorrected,
            x_order=self.variables_order,
            y_order=self.variables_order[::-1],
            label=labels[0])
        
        self.heatmap_pvalues(
            R['x'], R['y'], f, position, ax,
            size_scale=self.size_circles_big,
            alpha=pvalues['value'],
            x_order=self.variables_order,
            y_order=self.variables_order[::-1],
            label=labels[1])  

        leg = ax.legend(loc="upper right")

        leg.legend_handles[0].set_alpha(np.ones(len(pvalues.index))*self.alpha_pvalue_NotCorrected)
        leg.legend_handles[1].set_alpha(np.ones(len(pvalues.index)))
        leg.get_frame().set_linewidth(0.0)              
        leg.get_frame().set_facecolor([.92, .92, .92])

    def lower_triangular(self, data):
        
        """
        Extract lower triangular with no diagonal from square correlation matrix
        """
                
        df = pd.DataFrame(data=np.where(np.equal(*np.indices(data.shape)), np.nan, data.values), index=data.index, columns=data.columns) #NaN diagonal
        df = df.where(np.triu(np.ones(df.shape)).astype(bool))

        df = df.stack().reset_index()
        df.columns = ['x', 'y', 'value']
        
        return df

    def set_grid(self, f, position):
        
        GridSpec_width = 63
        
        plot_grid = plt.GridSpec(GridSpec_width, GridSpec_width, hspace=0.1, wspace=0.1) # Setup a 1x10 grid
        
        half = int( (GridSpec_width - 3) /2)
        
        if position == 0: #if colorbar is True
            ax = f.add_subplot(plot_grid[:half-3, :half-3]) # Use the left 14/15ths of the grid for the main plot
            
        elif position == 1:
            ax = f.add_subplot(plot_grid[:half-3, half+3:-3]) # Use the left 14/15ths of the grid for the main plot
            
        elif position == 2:
            ax = f.add_subplot(plot_grid[half+3:-3, :half-3]) # Use the left 14/15ths of the grid for the main plot

        elif position == 3:
            ax = f.add_subplot(plot_grid[half+3:-3, half+3:-3]) # Use the left 14/15ths of the grid for the main plot
            
        return ax, plot_grid

    def axis_heatmap(self, ax, x_to_num, y_to_num):
        
        x_to_num = {k:x_to_num[k] for k in list(x_to_num.keys())[:-1]}
        y_to_num = {k:y_to_num[k] for k in list(y_to_num.keys())[:-1]}
        
        ax.set_xticks([v+.5 for k,v in x_to_num.items()])
        ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
        ax.set_yticks([v for k,v in y_to_num.items()])
        ax.set_yticklabels([k for k in y_to_num])
        ax.tick_params(axis=u'both', which=u'both',length=0) #no ticks
        
        ax.grid(False, 'major') #turn off grid
        ax.grid(False, 'minor') #turn off grid
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()[:-1]], minor=True)
        
        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)     
        
        # Use lines to make stair-like lines, rather than full grid
        for i in x_to_num.values():
            ax.axhline(y=len(x_to_num)-1-i+.5, xmin =0, xmax=(i+1)/len(x_to_num), color=[.9, .9, .9])
            ax.axvline(x=len(x_to_num)-1-i+.5, ymin =0, ymax=(i+1)/len(x_to_num), color=[.9, .9, .9])
            
        for i in [3, 7, 11]: #highlight variable blocks
                
            ax.axhline(y=len(x_to_num)-1-i+.5, xmin =0, xmax=(i+1)/len(x_to_num), color=[.3, .3, .3])
            ax.axvline(x=len(x_to_num)-1-i+.5, ymin =0, ymax=(i+1)/len(x_to_num), color=[.3, .3, .3])


    def heatmap(self, x, y, f, clrb, position, experiment, **kwargs):
        
        """
        Code re-edited from:
        https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec    
        """
        
        # Color
        color = kwargs['color']
        palette = kwargs['palette']
        n_colors = len(palette)
        color_min, color_max = kwargs['color_range']

        def value_to_color(val):
    
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            
            return palette[ind]
    
        #Marker size
        size = kwargs['size']
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
        size_scale = kwargs.get('size_scale', 500)
    
        def value_to_size(val):
            
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            
            return val_position * size_scale
            
        #Variable names
        x_names = [t for t in kwargs['x_order']]     
        x_to_num = {p[1]:p[0] for p in enumerate(x_names)}
    
        y_names = [t for t in kwargs['y_order']]
        y_to_num = {p[1]:p[0] for p in enumerate(y_names)}
    
        ### Figure
    
        ax, plot_grid = self.set_grid(f, position)
        
        # Adjust the text style dynamically
        style = 'italic' if experiment == 'combined' else None
        
        ax.text(10, 10, experiment, weight='bold', style=style, fontsize=12)
        
        marker = kwargs.get('marker', 's')
    
        kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in ['color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order']}

        ax.scatter(
            x=[x_to_num[v] for v in x],
            y=[y_to_num[v] for v in y],
            marker=marker,
            s=[value_to_size(v) for v in size], 
            c=[value_to_color(v) for v in color],
            **kwargs_pass_on
        )
        
        self.axis_heatmap(ax, x_to_num, y_to_num)
    
        ######################################################
    
        if clrb:
            
            # Add color legend on the right side of the plot
            if color_min < color_max:
                
                if position == 0:
                    ax_colorbar = plt.subplot(plot_grid[:30,-1]) # Use the rightmost column of the plot
                elif position == 1:
                    ax_colorbar = plt.subplot(plot_grid[:30,-1]) # Use the rightmost column of the plot
                elif position == 3:
                    ax_colorbar = plt.subplot(plot_grid[33:63,-1])
                
                col_x = [0]*len(palette) # Fixed x coordinate for the bars
                bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
        
                bar_height = bar_y[1] - bar_y[0]
                ax_colorbar.barh(
                    y=bar_y,
                    width=[5]*len(palette), # Make bars 5 units wide
                    left=col_x, # Make bars start at 0
                    height=bar_height,
                    color=palette,
                    linewidth=0)
                
                ax_colorbar.set_xlim(1, 2) # lets crop the plot somewhere in the middle the colorbar
                ax_colorbar.grid(False) # Hide grid
                ax_colorbar.set_facecolor('white') # Make background white
                ax_colorbar.set_xticks([]) # Remove horizontal ticks
                ax_colorbar.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
                ax_colorbar.yaxis.tick_right() # Show vertical ticks on the right 
                ax_colorbar.set_title('R') #R values for colorbar #self.reliability_measure_label
                
                # Remove frame
                #ax.set_axis_off()
                ax_colorbar.spines['top'].set_visible(False)
                ax_colorbar.spines['right'].set_visible(False)
                ax_colorbar.spines['bottom'].set_visible(False)
                ax_colorbar.spines['left'].set_visible(False)
    
        return f, ax
    

    def heatmap_pvalues(self, x, y, f, position, ax, **kwargs):
        
        
        """
        Plot p-values as dots over the correlation squares 
        """
        
        alpha = kwargs['alpha']
        #s = kwargs['size']
        label=kwargs['label']
        size_scale = kwargs.get('size_scale', 500)
    
        x_names = [t for t in kwargs['x_order']]        
        x_to_num = {p[1]:p[0] for p in enumerate(x_names)}
    
        y_names = [t for t in kwargs['y_order']]
        y_to_num = {p[1]:p[0] for p in enumerate(y_names)}
    
        ###Figure
    
        ax.scatter(
            x=[x_to_num[v] for v in x],
            y=[y_to_num[v] for v in y],
            s=size_scale,
            edgecolors=['k' for v in alpha],
            alpha=[v for v in alpha],
            facecolors=['none' for v in alpha],
            label=label)
        
        ax.set_facecolor('none')
        
        self.axis_heatmap(ax, x_to_num, y_to_num)
        ax.patch.set_alpha(0)
        
class Figure5(Plot):
    
    def __init__(self, reliability_measure, method, extreme=False, split=False):
        
        super().__init__(method)
        
        self.reliability_measure = reliability_measure
        reliability_measure_dict = dict({'ICC': 'ICC', 'correlation': 'R'})
        self.reliability_measure_metric = reliability_measure_dict[reliability_measure]
        
        self.method = method
        self.extreme = extreme
        
        #pool_exp = False if not extreme else True
        pool_exp = False        
        #shuffled = True

        self.df_reliability_accuracy = self.fetch.df_reliability(reliability_measure, 'df_accuracy')
        self.df_reliability_accuracy = self.df_reliability_accuracy[[self.reliability_measure_metric]].T
        
        self.df_reliability_accuracy_simulations = self.fetch.df_reliability(reliability_measure, 'df_accuracy_simulations', method=method, extreme=extreme)      

        self.df_reliability_parameters = self.fetch.df_reliability(reliability_measure, 'df_parameters', pool_exp=pool_exp, method=method)
        self.df_reliability_parameters.rename(index=self.labels_to_replace, inplace=True)   
        self.df_reliability_parameters = self.df_reliability_parameters[[self.reliability_measure_metric]].T 
        
        self.df_reliability_parameters_simulations = self.fetch.df_reliability(reliability_measure, 'df_parameters_simulations', method=method, extreme=extreme)   
        self.df_reliability_parameters_simulations.rename(columns=self.labels_to_replace, inplace=True)
        self.df_reliability_parameters_simulations = self.df_reliability_parameters_simulations[self.df_reliability_parameters.columns]
                        
        self.df_slope_accuracy = self.fetch.df_slope('df_accuracy')
        self.df_slope_accuracy.columns.name = 'learning context'
        
        self.df_slopes_accuracy_simulations = self.fetch.df_slope('df_accuracy_simulations', method=method, pool_exp=pool_exp, extreme=extreme)
        self.df_slopes_accuracy_simulations.columns.name = 'learning context'

        self.df_slope_parameters = self.fetch.df_slope('df_parameters', pool_exp=pool_exp, method=method)
        self.df_slope_parameters.rename(columns=self.labels_to_replace, inplace=True)
        
        self.df_slopes_parameters_simulations = self.fetch.df_slope('df_parameters_simulations', method=method, pool_exp=pool_exp, extreme=extreme)
        self.df_slopes_parameters_simulations.rename(columns=self.labels_to_replace, inplace=True)
        
        ###
        
        if split:
        
            self.df_reliability_accuracy_simulations_top = self.fetch.df_reliability(reliability_measure, 'df_accuracy_simulations', method=method, extreme=extreme, split='top') 
            self.df_reliability_accuracy_simulations_bottom = self.fetch.df_reliability(reliability_measure, 'df_accuracy_simulations', method=method, extreme=extreme, split='bottom') 
        
            self.df_reliability_parameters_simulations_top = self.fetch.df_reliability(reliability_measure, 'df_parameters_simulations', method=method, extreme=extreme, split='top')
            self.df_reliability_parameters_simulations_top = self.df_reliability_parameters_simulations_top[self.df_reliability_parameters.columns] 
        
            self.df_reliability_parameters_simulations_bottom = self.fetch.df_reliability(reliability_measure, 'df_parameters_simulations', method=method, extreme=extreme, split='bottom')
            self.df_reliability_parameters_simulations_bottom = self.df_reliability_parameters_simulations_bottom[self.df_reliability_parameters.columns]
                
    def sim_vs_empirical(self, ax, df_R, df_R_sim, legend=False):
        
        x = df_R.columns.names[0]

        #order to variables on x axis
        order_x = self.learning_contexts if x == 'learning context' else df_R_sim.columns.tolist()
        
        #print(order_x) #check column order
        
        ax.axhline(y=0, color=self.colors.grey, linestyle='-', linewidth=0.7, alpha=.5) #zero line
        
        violins = sns.violinplot(data=df_R_sim, order=order_x, palette=self.colors.palette, inner=None, legend=False, ax=ax)
        
        if x == 'learning context':
            learning_contexts = [x for x in order_x if x in df_R_sim.columns and len(df_R_sim[x].unique()) > 1 ] if len(ax.collections) < len(order_x) else self.learning_contexts
        
        for i, violin in enumerate(ax.collections): #edit color of violins

            if x == 'learning context':              
                violin.set_alpha(.4), violin.set_facecolor(self.colors.learning_contexts[ learning_contexts[i] ])
                violin.set_edgecolor('k')
                
            else:
                violin.set_alpha(.4), violin.set_facecolor(self.colors.grey)
                
        ### Dotted distribution from simulations  
        sns.stripplot(data=df_R_sim, order=order_x, palette=[self.colors.grey]*len(order_x), alpha=.9, dodge=False, ax = ax) #label='Simulations' 
        
        ### Mean from simulation
        sns.stripplot(data=df_R_sim.mean().to_frame().T, order=order_x, marker='o', size=8, palette=['w']*len(order_x), edgecolor='k', linewidth=1, jitter=False, dodge=False, ax=ax, label='mean - synthetic data')
        
        ### True
        sns.stripplot(data=df_R, order=order_x, marker='o', size=8, palette=['k']*len(order_x), edgecolor='k', linewidth=1, jitter=False, dodge=False, ax=ax, label='empirical data')

        #Axis labels
        ax.set_xlabel('')
        
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
                
        ### Legend
        
        if legend:
            #breakpoint()
            #h, l = ax.get_legend_handles_labels()
            #leg_plot = [0, 4, 8] #select which labels to report in legend (#0 for dot marker)
            #h_idx = [h[index] for index in leg_plot] #get coloured marker for selected labels
            #l_idx = [l[index] for index in leg_plot] #get text for selected labels
            #plt.legend(h_idx, l_idx, loc='lower right') #plot selected labels
            #ax.legend().set_visible(True)
            
            # Collect handles and labels for the legend
            handles, labels = ax.get_legend_handles_labels()

            # Deduplicate legend entries by creating a dictionary, preserving order
            unique_legend = dict(zip(labels, handles))

            # Add the legend back with unique entries
            leg = ax.legend(unique_legend.values(), unique_legend.keys(), facecolor=[.9, .9, .9], framealpha=1, loc='upper center', bbox_to_anchor=(0.665, 1.335), title='test-retest reliability', title_fontproperties={'weight': 'bold', 'style':'italic'}, handletextpad=0.1)
            leg.get_frame().set_linewidth(0.0)
            
        else:
            
            ax.legend().set_visible(False)
            
    
    def panels(self, ax):
        
        bbox = {'facecolor': self.colors.grey,
                'edgecolor': [.2, .2, .2],
                'alpha':0.4,
                'pad':6}
        
        bbox_side = {'facecolor': 'white',
                'edgecolor': 'white',
                'alpha':0.4,
                'pad':6}
    
        y_offset = .12
        
        if not self.extreme:
        
            ax.text(x=0, y=1-y_offset, s='Data', bbox=bbox_side)
    
            ax.text(x=.45, y=1-y_offset, s='Test \n \n (RP, RC, PP, PC) \n x (N=169)', bbox =bbox, ha='center', va='center')
            ax.text(x=.85, y=1-y_offset, s='Retest \n \n (RP, RC, PP, PC) \n x (N=169)', bbox =bbox, ha='center', va='center')
        
            ax.annotate("", xy=(.65, .83-y_offset), xytext=(.85, .92-y_offset), arrowprops=dict(arrowstyle="->"))    
            ax.annotate("", xy=(.65, .83-y_offset), xytext=(.45, .92-y_offset), arrowprops=dict(arrowstyle="->"))
        
        ax.text(x=0, y=.75-y_offset, s='Parameters', bbox =bbox_side)
        ax.text(x=.65, y=.75-y_offset, s='Model parameters \n \n '+r'$\beta$, '+r'$\alpha_{v}$, '+r'$\alpha_{+}$, '+r'$\alpha_{-}$'+' \n x (N=169)', bbox =bbox, ha='center', va='center')
     
        ax.annotate("", xy=(.85, .60-y_offset), xytext=(.65, .67-y_offset), arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(.45, .60-y_offset), xytext=(.65, .67-y_offset), arrowprops=dict(arrowstyle="->"))
        
        ax.text(x=0, y=.5-y_offset, s='Simulations', bbox =bbox_side)   
        ax.text(x=.45, y=.5-y_offset, s='Test \n \n (RP, RC, PP, PC) \n x (N=169) \n x (n=100)', bbox =bbox, ha='center', va='center')
        ax.text(x=.85, y=.5-y_offset, s='Retest \n \n (RP, RC, PP, PC) \n x (N=169) \n x (n=100)', bbox =bbox, ha='center', va='center')
        
        ax.annotate("", xy=(.45, .3-y_offset), xytext=(.45, .38-y_offset), arrowprops=dict(arrowstyle="->"))   
        ax.annotate("", xy=(.85, .3-y_offset), xytext=(.85, .38-y_offset), arrowprops=dict(arrowstyle="->"))
        
        ax.text(x=0, y=.2-y_offset, s='Parameters', bbox =bbox_side)  
        ax.text(x=.45, y=.2-y_offset, s='Model parameters \n \n '+r'$\beta$, '+r'$\alpha_{v}$, '+r'$\alpha_{+}$, '+r'$\alpha_{-}$'+' \n x (N=169) \n x (n=100)', bbox =bbox, ha='center', va='center')
        ax.text(x=.85, y=.2-y_offset, s='Model parameters \n \n '+r'$\beta$, '+r'$\alpha_{v}$, '+r'$\alpha_{+}$, '+r'$\alpha_{-}$'+' \n x (N=169) \n x (n=100)', bbox =bbox, ha='center', va='center')

    
    def plot(self):
        
        mosaic = [['left', '.', 'upper centre', 'upper right'],
                  ['left', '.', 'lower centre', 'lower right']]
        
        gs_kw = dict(width_ratios=[1, .2, 1, 1], height_ratios=[1, 1])
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(11.2, 4.75), gridspec_kw=gs_kw)
    
        ### Left
        self.panels(axd['left'])
        axd['left'].axis("off")
    
        ### Central column
        
        axd['upper centre'].set_ylabel(f'correlation ({self.reliability_measure_metric})')
        axd['lower centre'].set_ylabel('regression slope (m)') #slope from linear model retest = m x + c
    
        self.sim_vs_empirical(axd['upper centre'], self.df_reliability_accuracy, self.df_reliability_accuracy_simulations, legend=True)    
        self.sim_vs_empirical(axd['lower centre'], self.df_slope_accuracy, self.df_slopes_accuracy_simulations, legend=True)
        
        ### Right column
        self.sim_vs_empirical(axd['upper right'], self.df_reliability_parameters, self.df_reliability_parameters_simulations, legend=True)
        self.sim_vs_empirical(axd['lower right'], self.df_slope_parameters, self.df_slopes_parameters_simulations, legend=True)
        
        #Limits, labels, title, layout
        if not self.extreme:
        
            y_lim_reliability = [-.16, 1]
            axd['upper centre'].set_ylim(y_lim_reliability)
            axd['upper right'].set_ylim(y_lim_reliability)
            
            y_lim_reliability = [-.16, 1.1]
            axd['lower centre'].set_ylim(y_lim_reliability)
            axd['lower right'].set_ylim(y_lim_reliability)
            
        else:
            
            axd['upper centre'].set_ylim([-1, 1])
            axd['upper right'].set_ylim([-1, 1])
            axd['lower centre'].set_ylim([-1.1, 1.1])
            axd['lower right'].set_ylim([-1.1, 1.1])
    
        #x tick labels
        plt.setp(axd['upper centre'].get_xticklabels(), visible=False)
        plt.setp(axd['upper right'].get_xticklabels(), visible=False)
    
        #y tick labels
        #plt.setp(axd['upper right'].get_yticklabels(), visible=False)
        #plt.setp(axd['lower right'].get_yticklabels(), visible=False)

        axd['upper right'].tick_params(axis='y', which='both', labelleft=False)
        axd['lower right'].tick_params(axis='y', which='both', labelleft=False)
 
        ### Other details
    
        plt.tight_layout()
        fig.subplots_adjust(wspace=.13, hspace=.35)
    
        plt.figtext(0, .95, string.ascii_uppercase[0]+'.', weight='bold', fontsize=13)
        plt.figtext(.355, .95, string.ascii_uppercase[1]+'.', weight='bold', fontsize=13)
        plt.figtext(.355, .46, string.ascii_uppercase[3]+'.', weight='bold', fontsize=13)
        plt.figtext(.6875, .95, string.ascii_uppercase[2]+'.', weight='bold', fontsize=13)
        plt.figtext(.6875, .46, string.ascii_uppercase[4]+'.', weight='bold', fontsize=13)
        
        title = f'Fig5 - {self.reliability_measure} - RL parameter fit by {self.method}'
        title += ' - extreme' if self.extreme else ''
        self.save.figure(title)
        
        return
    
    def plot_split(self):
        
        mosaic = [['left', '.', 'upper centre', 'upper right'],
                  ['left', '.', 'lower centre', 'lower right']]
        
        gs_kw = dict(width_ratios=[1, .2, 1, 1], height_ratios=[1, 1])
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 6), gridspec_kw=gs_kw)
    
        ### Left
        self.panels(axd['left'])
        axd['left'].axis("off")
    
        ### Central column
        
        axd['upper centre'].set_ylabel(f'{self.reliability_measure_metric}')
        axd['lower centre'].set_ylabel(f'{self.reliability_measure_metric}') #slope from linear model retest = m x + c
    
        self.sim_vs_empirical(axd['upper centre'], self.df_reliability_accuracy, self.df_reliability_accuracy_simulations_top)    
        self.sim_vs_empirical(axd['lower centre'], self.df_reliability_accuracy, self.df_reliability_accuracy_simulations_bottom)
        
        ### Right column
        self.sim_vs_empirical(axd['upper right'], self.df_reliability_parameters, self.df_reliability_parameters_simulations_top, legend=True)
        self.sim_vs_empirical(axd['lower right'], self.df_reliability_parameters, self.df_reliability_parameters_simulations_bottom)
        
        #Limits, labels, title, layout
        if not self.extreme:
            axd['upper centre'].set_ylim([-.25, 1])
            axd['upper right'].set_ylim([-.25, 1])
            axd['lower centre'].set_ylim([-.25, 1.1])
            axd['lower right'].set_ylim([-.25, 1.1])
        else:
            axd['upper centre'].set_ylim([-1, 1])
            axd['upper right'].set_ylim([-1, 1])
            axd['lower centre'].set_ylim([-1.1, 1.1])
            axd['lower right'].set_ylim([-1.1, 1.1])
    
        #x tick labels
        plt.setp(axd['upper centre'].get_xticklabels(), visible=False)
        plt.setp(axd['upper right'].get_xticklabels(), visible=False)
    
        #y tick labels
        #plt.setp(axd['upper right'].get_yticklabels(), visible=False)
        #plt.setp(axd['lower right'].get_yticklabels(), visible=False)

        axd['upper right'].tick_params(axis='y', which='both', labelleft=False)
        axd['lower right'].tick_params(axis='y', which='both', labelleft=False)
 
        ### Other details
    
        plt.tight_layout()
        fig.subplots_adjust(wspace=.1, hspace=.1)
    
        plt.figtext(0, 1, string.ascii_uppercase[0]+'.', weight='bold', fontsize=13)
        plt.figtext(.355, 1, string.ascii_uppercase[1]+'.', weight='bold', fontsize=13)
        plt.figtext(.355, .49, string.ascii_uppercase[3]+'.', weight='bold', fontsize=13)
        plt.figtext(.6875, 1, string.ascii_uppercase[2]+'.', weight='bold', fontsize=13)
        plt.figtext(.6875, .49, string.ascii_uppercase[4]+'.', weight='bold', fontsize=13)
        
        title = f'Fig5 - {self.reliability_measure} - RL parameter fit by {self.method}'
        title += ' - extreme with split' if self.extreme else ' - split'
        self.save.figure(title)
        
        return    

class Figure5CompareNSessions(Plot):
    
    def __init__(self, reliability_measure, method):
        
        super().__init__(method)
        
        self.reliability_measure = reliability_measure
        reliability_measure_dict = dict({'ICC': 'ICC', 'correlation': 'R'})
        self.reliability_measure_metric = reliability_measure_dict[reliability_measure]
        
        self.method = method
        pool_exp = False

        self.df_reliability_parameters = self.fetch.df_reliability(reliability_measure, 'df_parameters', pool_exp=pool_exp, method=method)
        self.df_reliability_parameters = self.df_reliability_parameters[[self.reliability_measure_metric]].T
        
        n_sessions_list = [1, 2, 4]
        df_reliability_parameters_simulations_list = []

        for n_sessions in n_sessions_list:
            
            df_reliability_parameters_simulations = self.fetch.df_reliability(reliability_measure, 'df_parameters_simulations', method=method, n_sessions=n_sessions)
            df_reliability_parameters_simulations['N sessions'] = n_sessions
            df_reliability_parameters_simulations_list.append(df_reliability_parameters_simulations)

        self.df_reliability_parameters_simulations = pd.concat(df_reliability_parameters_simulations_list)

        self.df_reliability_parameters_simulations = self.df_reliability_parameters_simulations.set_index('N sessions')[self.df_reliability_parameters.columns]

    def sim_vs_empirical(self, ax, df_R, df_R_sim, col, col_n, legend=False):
                        
        ax.axhline(y=0, color=self.colors.grey, linestyle='-', linewidth=0.7, alpha=.5) #zero line
        ax.axhline(y=df_R[col].values, color='k', linestyle='-', linewidth=1, label='Empirical') #empirical

        sns.violinplot(data=df_R_sim, x='N sessions', y=col, inner=None, legend=False, ax=ax)
            
        for i, violin in enumerate(ax.collections): #edit color of violins
            violin.set_alpha(.4), violin.set_facecolor(self.colors.grey)

        Nsessions = len(df_R_sim.index.unique())
                
        ### Dotted distribution from simulations  
        sns.stripplot(df_R_sim, x='N sessions', y=col, palette=[self.colors.grey]*Nsessions, alpha=.9, dodge=False, ax = ax) 
        
        #breakpoint()
        ### Mean from simulation
        sns.stripplot(data=df_R_sim.groupby('N sessions').mean().reset_index(), x='N sessions', y=col, marker='o', size=8, palette=['w']*Nsessions, edgecolor='k', linewidth=1, jitter=False, dodge=False, ax=ax, label='Simulations (mean)')
            
        #Axis labels
        ax.set_xlabel('') if col_n < 2 else None 
        ax.set_ylabel(self.reliability_measure) if col_n % 2 == 0 else ax.set_ylabel('') 
        
        #ax.spines['bottom'].set_visible(False)
        #ax.xaxis.set_ticks_position('none')

        ax.set_title(col)
        ax.set_ylim([-1, 1])
        
        ### Legend
        
        if col_n == 1:
            #breakpoint()
            #h, l = ax.get_legend_handles_labels()
            #leg_plot = [0, 4, 8] #select which labels to report in legend (#0 for dot marker)
            #h_idx = [h[index] for index in leg_plot] #get coloured marker for selected labels
            #l_idx = [l[index] for index in leg_plot] #get text for selected labels
            #plt.legend(h_idx, l_idx, loc='lower right') #plot selected labels
            #ax.legend().set_visible(True)
            
            # Collect handles and labels for the legend
            handles, labels = ax.get_legend_handles_labels()

            # Deduplicate legend entries by creating a dictionary, preserving order
            unique_legend = dict(zip(labels, handles))

            # Add the legend back with unique entries
            ax.legend(unique_legend.values(), unique_legend.keys())
            
        else:
            
            ax.legend().set_visible(False)

    def plot(self):
        
        fig, axd = plt.subplots(2, 2, figsize=(8,8))
        axd = axd.flatten()
                
        #axd['upper left'].set_ylabel(f'{self.reliability_measure_metric}')
        #axd['lower left'].set_ylabel(f'{self.reliability_measure_metric}')

        columns = self.df_reliability_parameters.columns

        [self.sim_vs_empirical(axd[col_n], self.df_reliability_parameters, self.df_reliability_parameters_simulations, col, col_n, legend=True) for col_n, col in enumerate(columns)]
        
        #Limits, labels, title, layout

        #x tick labels
        #plt.setp(axd['upper left'].get_xticklabels(), visible=False)
        #plt.setp(axd['upper right'].get_xticklabels(), visible=False)
    
        #axd[1].tick_params(axis='y', which='both', labelleft=False)
        #axd[3].tick_params(axis='y', which='both', labelleft=False)
 
        ### Other details
    
        fig.subplots_adjust(wspace=.2, hspace=.2)
        plt.tight_layout()
    
        plt.figtext(0, 1, string.ascii_uppercase[0]+'.', weight='bold')
        plt.figtext(.5, 1, string.ascii_uppercase[1]+'.', weight='bold')
        plt.figtext(0, .5, string.ascii_uppercase[2]+'.', weight='bold')
        plt.figtext(.5, .5, string.ascii_uppercase[3]+'.', weight='bold')
        
        title = f'Fig5 compare N sessions - {self.reliability_measure} - RL parameter fit by {self.method}'
        self.save.figure(title)    


class ICCBootstrap(Plot):

    def __init__(self, folder):

        super().__init__(folder)

        self.folder = folder

    def plot(self, method):

        df = self.fetch.df_icc_bootstrap(method)
        df_mean = self.fetch.df_icc_bootstrap_mean(method)

        plt.figure(figsize=(6, 6))
        ax = sns.violinplot(data=df, x='parameter', y='ICC', inner=None, legend=False, color=[.5, .5, .5])
        [violin.set_alpha(.4) for i, violin in enumerate(ax.collections)]
        sns.stripplot(data=df, x='parameter', y='ICC', ax = ax, color='k', alpha=.1)
        sns.stripplot(data=df_mean, x='parameter', y='ICC', ax = ax, color='w')
        ax.set_ylim([0, 1])
        #ax.set_ylim([-1, 1])

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
            
        self.save.figure(f'ICC_boostrap_{method}')

    def prepare_df(self, method):

        super().__init__(method) #change method data folder

        df = self.fetch.df_icc_bootstrap(method)
        df['method'] = self.method_dict[method]

        return df

    def prepare_df_mean(self, method):

        super().__init__(method) #change method data folder

        df_mean = self.fetch.df_icc_bootstrap_mean(method)
        df_mean['method'] = method

        return df_mean

    def plot_combined(self, method_list):
        
        df_list = [self.prepare_df(method) for method in method_list]
        df_mean_list = [self.prepare_df_mean(method) for method in method_list]

        super().__init__(self.folder)

        df = pd.concat(df_list)
        df_mean = pd.concat(df_mean_list)

        palette = ['k', 'r', 'c', 'b']

        plt.figure(figsize=(12, 6))
        
        ax = sns.violinplot(data=df, x='parameter', y='ICC', hue='method', inner=None, legend=True, palette=palette, alpha=.3)
        #[violin.set_alpha(.4) for i, violin in enumerate(ax.collections)]
        sns.stripplot(data=df, x='parameter', y='ICC', hue='method', dodge=True, ax = ax, palette = ['k', 'k', 'k', 'k'], alpha=.1, legend=False)
        sns.stripplot(data=df_mean, x='parameter', y='ICC', hue='method', dodge=True, ax = ax, palette=['w', 'w', 'w', 'w'], legend=False, edgecolor='k', linewidth=1)
        #ax.set_ylim([0, 1])
        ax.set_ylim([-1, 1]), ax.axhline(y=0, color=[.5, .5, .5], alpha=.3)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
            
        self.save.figure(f'ICC_boostrap_combined')

class TotalVarianceExplained(Plot):

    def __init__(self, folder):

        super().__init__(folder)

        self.folder = folder

        self.parameter_names = list( self.fetch.greek_to_plain_dict.keys() )
        self.propensity_variable_names = list( self.fetch.propensity_dict.values() )
        self.clinics_variables_names = list( self.fetch.clinics_dict.values() )
        self.questionnaire_variables_names = self.propensity_variable_names + self.clinics_variables_names

    def plot(self, exp, method_list):

        df_list = [self.prepare_df(exp, method) for method in method_list]
        df = pd.concat(df_list)

        super().__init__(self.folder)

        clinics_color = '#008080' #as in Fig. 4
        propensity_color = '#0b5394'

        plt.figure(figsize=(8, 4))
        
        #Detailed
        #df.set_index(['method', 'measure class', 'index', 'level_1']).unstack(['measure class','index', 'level_1']).plot(kind='bar', stacked=True)
        
        df = df.drop(['index', 'level_1'], axis=1).groupby(['method', 'measure class']).sum().unstack('measure class')[r'$R^2$']
        
        mapping = {method: i for i, method in enumerate(method_list)}
        key = df.index.map(mapping)
        df = df.iloc[key.argsort()] #reorder index
        
        #Summary
        ax = df.plot.bar(stacked=True, color=[clinics_color, propensity_color])
        ax.set_ylabel(r'$R^2$')
        #ax = sns.violinplot(data=df, x='method', y=r'$R^2$', color='k', legend=False, alpha=.3, inner=None, hue_order=method_list)
        #sns.stripplot(data=df, x='method', y=r'$R^2$', color='k', legend=False, ax=ax, alpha=.8, hue_order=method_list)
        #sns.stripplot(data=df.groupby('method').mean(), x='method', y=r'$R^2$', ax=ax, color='w', edgecolor='k', linewidth=1, size=10, hue_order=method_list)

        ax.set_ylim([0, None])
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
            
        self.save.figure(f'Total_variance_explained_by_methods_exp_{exp}')


    def prepare_df(self, exp, method):

        super().__init__(method) #change method data folder
        
        df = self.fetch.df_cross_correlation_dict(experiment=exp, method=method, p=False)['R']
        df = df[self.parameter_names].loc[self.questionnaire_variables_names].stack()
        df = df**2 #squaring R-values
        df = df.to_frame()
        df = df.rename(columns={0: r'$R^2$'})
        df.reset_index(inplace=True)
        df['method'] = method
        df['measure class'] = df['index'].apply(lambda x: 'propensity' if x in self.propensity_variable_names else 'clinics' if x in self.clinics_variables_names else None )

        self.save.export(df.reset_index(), self.save.folder_cross_correlation, f'df_tot_variance_explained_exp_{exp}_method_{method}')

        return df

class Fig4CompareMethods(Plot):

    def __init__(self, folder, reliability_measure):

        super().__init__(folder)

        self.folder = folder
        self.reliability_measure = reliability_measure
        
    def class_measure(self, df, color, label, ax):
        
        reliability_measure = 'R' if self.reliability_measure == 'correlation' else self.reliability_measure
        
        df.plot.scatter(reliability_measure, 'CV', marker='s', color=color, label=label, ax = ax)

        #add annotations
        text_x, text_y = np.zeros(4), np.zeros(4)
        [plt.annotate(df.index[idx_n], xy=xy, xytext=(xtxt, ytxt), textcoords="offset points") for idx_n, (xy, xtxt, ytxt) in enumerate(zip(zip(df[f'{reliability_measure}'], df.CV), text_x, text_y))]       

    def reliability_vs_CV(self):

        method = 'MAP'
        super().__init__(method) #change method data folder
        self.df_accuracy = self.fetch.df_reliability_vs_CV(self.reliability_measure, f'df_accuracy')
        self.df_propensity = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_propensity')
        self.df_clinics = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_clinics')
        self.df_parameters_MAP = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_parameters', method=method)
        
        method = 'HB'
        super().__init__(method) #change method data folder
        self.df_parameters_HB = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_parameters', method=method)

        method = 'HBpool'
        super().__init__(method) #change method data folder        
        self.df_parameters_HBpool = self.fetch.df_reliability_vs_CV(self.reliability_measure, 'df_parameters', method=method)

        super().__init__(self.folder)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
               
        ### Accuracy
        self.class_measure(self.df_accuracy, '#cc0000', 'Behavioural measures', ax=ax)
       
        ### Parameters
        self.class_measure(self.df_parameters_MAP, '#ff9933', label='Computational measures (MAP)', ax=ax)
        self.class_measure(self.df_parameters_HB, '#ffd833', label='Computational measures (HB)', ax=ax)
        self.class_measure(self.df_parameters_HBpool, '#deff33', label='Computational measures (HB pool)', ax=ax)
      
        ### Clinical
        self.class_measure(self.df_clinics, '#008080', 'Clinical measures', ax=ax)
      
        ### Propensity
        self.class_measure(self.df_propensity, '#0b5394', 'Propensity measures', ax=ax)
       
        plt.xlabel(f'Test-retest {self.reliability_measure}')
        plt.ylabel('Coefficient of variation')
   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
   
        ax.set_xlim([0, 1])
       
        leg = ax.legend(handletextpad=0.1)
        leg.get_frame().set_linewidth(0.0)
        
        self.save.figure(f'Reliability_vs_CV_compare_methods - {self.reliability_measure}')

class FigureReactionTime(Plot):
    
    def __init__(self, folder):
        
        super().__init__(folder)
        
        df = self.fetch.df()
        self.df = df.groupby(['exp', 'learning context', 'id']).median()['reaction time'].rename('Reaction time').to_frame()
        self.df.reset_index(inplace=True)
        self.df['exp'] = self.df['exp'].map(self.exp_dict)
        
    def plot(self):
        
        #g = sns.FacetGrid(data=self.df, hue="exp", col="learning context", col_wrap=4, sharex=True)
        #g.map(plt.hist, "Reaction time", alpha=.4)
        #plt.legend(loc='upper right', fancybox=True, fontsize=8)
        #plt.show()
        
        g = sns.displot(data=self.df, x='Reaction time', hue="exp", col="learning context", col_wrap=4, col_order=self.learning_contexts, height=2, aspect=1, palette=[[.5, 0, 0], [0, 0, .5]], alpha=.85)
        g.set_titles("{col_name}")
        g.set_xlabels('Reaction time (s)')

        title = 'Fig reaction time'
        self.save.figure(title)

class SummaryStats(Plot):

    def __init__(self, folder):
        
        super().__init__(folder)
        
        self.method = folder

    def describe(self, df_name, method=None):
    
        if df_name != 'df_parameters':
            df = self.fetch.df_dict[df_name]()
        elif df_name == 'df_parameters':
            df = self.fetch.df_dict[df_name](method=self.method, pool_exp=False)
        
        for key in [0, 1]:
            
            df_temp = df.xs(level='exp', key=key)
            df_described = df_temp.describe()
            
            #Add S.E.M.
            df_described = pd.concat([df_described, df.xs(level='exp', key=key).sem().to_frame().rename(columns={0:'sem'}).T])
            
            self.save.export(df_described.reset_index(), self.save.folder_stats_summary, f'{df_name}_description_stats_exp_{key}')
        
        df_temp = df.unstack('exp')
        
        columns = ['t', 'p', 'diff_mean', 'diff_SEM']
        df_ttest = pd.DataFrame(columns=columns)
        
        for variable in df.columns:
            res = ttest_rel(df_temp[variable][1], df_temp[variable][0])
            mean_diff = (df_temp[variable][1]-df_temp[variable][0]).mean()
            sem_diff = (df_temp[variable][1]-df_temp[variable][0]).sem()
            data = [res.statistic, res.pvalue, mean_diff, sem_diff]
            df_ttest_temp = pd.DataFrame(columns=columns, data=[data], index=[variable])
            df_ttest = pd.concat([df_ttest, df_ttest_temp])
        
        self.save.export(df_ttest.reset_index(), self.save.folder_ttests, f'{df_name}_ttest')
 
    def reliability_simulations(self, reliability_measure, method=None, extreme=False): 

        df_reliability_parameters_simulations = self.fetch.df_reliability(reliability_measure, 'df_parameters_simulations', method=method, extreme=extreme)   
        #df_reliability_parameters_simulations = df_reliability_parameters_simulations[df_reliability_parameters.columns]
        
        df_described = df_reliability_parameters_simulations.describe()
        
        #Add S.E.M.
        df_described = pd.concat([df_described, df_reliability_parameters_simulations.sem().to_frame().rename(columns={0:'sem'}).T])

        self.save.export(df_described.reset_index(), self.save.folder_stats_summary, f'df_reliability_parameters_simulations_description_stats_{reliability_measure}')

    def ttest_alpha_con_disc(self, method=None, extreme=False): 

        df_name = 'df_parameters'

        df = self.fetch.df_dict[df_name](method=self.method, pool_exp=False)
        
        df_temp = df.unstack('exp')
        columns = ['t', 'p', 'diff_mean', 'diff_SEM']        
        variable1, variable2 = r'$\alpha_{CON}$', r'$\alpha_{DISC}$'
        
        for experiment in [0, 1]:
        
            df_ttest = pd.DataFrame(columns=columns)
            res = ttest_rel(df_temp[variable1][experiment], df_temp[variable2][experiment])
            mean_diff = (df_temp[variable1][experiment]-df_temp[variable2][experiment]).mean()
            sem_diff = (df_temp[variable1][experiment]-df_temp[variable2][experiment]).sem()
            data = [res.statistic, res.pvalue, mean_diff, sem_diff]
            df_ttest_temp = pd.DataFrame(columns=columns, data=[data], index=[variable1+'-'+variable2])
            df_ttest = pd.concat([df_ttest, df_ttest_temp])
        
            self.save.export(df_ttest.reset_index(), self.save.folder_ttests, f'{df_name}_alpha_con_disc_exp_{experiment}_ttest')
        
class DemographicsTable(Plot):
    
    def __init__(self, folder):
        
        super().__init__(folder)
        
    def general(self):        
        self.fetch.df_demographics()
        print(f'See folder {self.save.folder_demographics}')        
        
    def approved(self):        
        self.fetch.df_demographics_approved()
        print(f'See folder {self.save.folder_demographics_approved}')                
