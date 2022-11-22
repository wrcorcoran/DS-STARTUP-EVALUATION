import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE =\
["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

file_path = '/Users/willcorcoran__/Documents/data-science-proj/startup-proj/CAX_Startup_Data.csv'
df = pd.read_csv(file_path, encoding='latin1')
cols = ['Dependent-Company Status',
        'year of founding',
        'Employee Count',
        'Employees count MoM change',
        'Last Funding Amount',
        # 'Country of company',
        'Number of Investors in Seed',
        'Number of Investors in Angel and or VC',
        'Number of Co-founders',
        'Team size all employees',
        'Presence of a top angel or venture fund in previous round of investment',
        'Number of of repeat investors',
        'Worked in top companies',
        'Have been part of successful startups in the past?',
        'Average Years of experience for founder and co founder',
        'Highest education',
        'Relevance of education to venture',
        'Degree from a Tier 1 or Tier 2 university?',
        'Skills score',
        'Industry trend in investing',
        'Disruptiveness of technology',
        'Number of Direct competitors',
        'Employees per year of company existence',
        'Last round of funding received (in milionUSD)',
        'Time to 1st investment (in months)',
        'Avg time to investment - average across all rounds, measured from previous investment',
        'Percent_skill_Entrepreneurship',
        'Percent_skill_Operations',
        'Percent_skill_Engineering',
        'Percent_skill_Marketing',
        'Percent_skill_Leadership',
        'Percent_skill_Data Science',
        'Percent_skill_Business Strategy',
        'Percent_skill_Product Management',
        'Percent_skill_Sales',
        'Percent_skill_Domain',
        'Percent_skill_Law',
        'Percent_skill_Consulting',
        'Percent_skill_Finance',
        'Percent_skill_Investment',
        'Renown score']

df = df[cols]
df = df.replace({'No Info': None, 'Success': 1, 'Failed': 0, 'No': 0, 'Yes': 1, 'Bachelors': 1,
                 'Masters': 2, 'PhD': 3, 'None': 0, 'Both': 3, 'Tier_1': 1, 'Tier_2': 2,
                 'Low': 0, 'Medium': 1, 'High': 2}, inplace=False)

df = df.dropna(how='any')
print(df.head())

x = np.array(df[cols[1:]], dtype=np.float16)
y = df['Dependent-Company Status']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

X_train = torch.from_numpy(X_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

