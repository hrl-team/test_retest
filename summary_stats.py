

#Maybe for saftey it would be great if you can produce summary stats tables for most empirical ouputs (e.g. choice accuracy, RTs, model parameters etc. --- we would need means+/- sem for both test and re-test, stats (tval, pval) for ttest of difference between test and re-test, and test-retest reliability metrics)

import pandas as pd
from scipy.stats import ttest_ind

df = 

df.describe()

#define samples
group1 = df[df['exp']==0]
group2 = df[df['exp']==1]

#perform independent two sample t-test
ttest_ind(group1['score'], group2['score'])
