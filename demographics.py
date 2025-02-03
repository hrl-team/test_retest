#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:36:01 2023

@author: stefanovrizzi
"""

import pandas as pd
import numpy as np

class Demographics:        

    def __init__(self, save):
        
        self.save = save
 
    def pivot(self, data, experiment):

        if experiment == 'test':
            
            demographics_of_interest = ['Participant id', 'status', 'Age', 'sex', 'ethnicity', 'Ethnicity 2', 'Ethnicity (Simplified UK/US Census)']
            testA = data[0][demographics_of_interest]
            testB = data[1][demographics_of_interest]
            
            df = pd.concat([testA, testB])
            df.rename(columns={'Ethnicity (Simplified UK/US Census)': 'Ethnicity (Simplified UK US Census)'}, inplace=True) #avoid '/' to save file correctly
            
        elif experiment == 'retest':
            demographics_of_interest = ['participant_id', 'status', 'age', 'Sex', 'Current Country of Residence', 'Nationality', 'Country of Birth']
            df = data[demographics_of_interest]
            df = df.rename(columns={'participant_id': 'participant id'})
            
        df.columns = df.columns.str.lower()
        
        #df.isna().sum() #find 'nan' in dataframe
        df['sex'] = df['sex'].fillna('NA')
        df['age range'] = df['age'].apply(lambda x: '18-30' if x <= 30 else '30-60' if x > 30 and x <= 60 else '60+' if x > 60 else None)
        
        self.save.export(df.reset_index(), self.save.folder_demographics, f'demographics_{experiment}')
        
        columns_of_interest = df.columns[ df.columns.str.contains('|'.join(['ethnicity', 'nationality', 'country']), case=False) ]
        
        for ethnicity_classification in columns_of_interest:

            demographics_test = df.groupby(['status', 'sex', 'age range', ethnicity_classification])['participant id'].count().to_frame().rename(columns={'participant id': 'counts'})
            self.save.export(demographics_test.reset_index(), self.save.folder_demographics, f'demographics_{experiment}_counts_{ethnicity_classification}')
                   
    def participation_joint(self, test, retest):
        
        test.set_index('participant id', inplace=True)
        test['exp'] = 'test'
        retest.set_index('participant id', inplace=True)
        retest['exp'] = 'retest'
        
        df = pd.concat([test, retest])
        self.save.export(df.reset_index(), self.save.folder_demographics, 'participants')
        
        participants = df.reset_index().groupby(['exp', 'status'])['participant id'].count().to_frame().rename(columns={'participant id': 'counts'})
        self.save.export(participants.reset_index(), self.save.folder_demographics, 'participants_count')

        participants = df.reset_index().groupby(['exp', 'status', 'sex', 'age range'])['participant id'].count().to_frame().rename(columns={'participant id': 'counts'})
        self.save.export(participants.reset_index(), self.save.folder_demographics, 'participants_count_detailed')
        
    def time_window_between_exp(self, test1, retest):
        
        retest = retest[retest['status']=='Approved']
        retest[['completed_date_time', 'New']] = retest['completed_date_time'].str.split(' ', expand=True)
        retest = retest[['participant_id', 'completed_date_time', 'age', 'Sex']]
        
        test1.set_index('Participant id', inplace=True)
        retest.set_index('participant_id', inplace=True)
        
        result = pd.concat([test1, retest], axis=1, join="inner")
        result[['completed_date_time', 'Completed Date Time']] = result[['completed_date_time', 'Completed Date Time']].apply(pd.to_datetime)
        result['diff_days'] = (result['completed_date_time'] - result['Completed Date Time']) / np.timedelta64(1, 'D')
        
        sex = {'Female' : 1,'Male' : 0}
        result.sex = result.sex.map(sex)
        
        print(f'# subjects {len(result)}')
        print()
        print('Time difference retest')
        print(f"mean {result.diff_days.mean()}")
        print(f"SD {result.diff_days.std()}")
        print(f"SEM {result.diff_days.sem()}")
        print(f"max {result.diff_days.max()}")
        print(f"min {result.diff_days.min()}")
        print()
        print('Age test') #In the data matrices of exp 1, Age is with capital letter
        print(f"mean {result.Age.mean()}")
        print(f"SD {result.Age.std()}")
        print(f"SEM {result.Age.sem()}")
        print(f"max {result.Age.max()}")
        print(f"min {result.Age.min()}")
        print()
        print('Age retest') #In the data matrices of exp 2, age is with small letter
        print(f"mean {result.age.mean()}")
        print(f"SD {result.age.std()}")
        print(f"SEM {result.age.sem()}")
        print(f"max {result.age.max()}")
        print(f"min {result.age.min()}")
        print()
        print('Sex')
        print(f"mean {result.sex.mean()}")
        print(f"SD {result.sex.std()}")
        print(f"SEM {result.sex.sem()}")