import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

use_cuda = torch.cuda.is_available()
# torch.autograd.set_detect_anomaly(True)
# np.seterr(all="raise")

class __Dataset(data.Dataset):
        def __init__(self):
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
                df = df.replace({'No Info': None, 'Success': 1, 'Failed': 0, 'No': 0, 'Yes': 1,
                                 'Bachelors': 1,
                                 'Masters': 2, 'PhD': 3, 'Both': 3, 'Tier_1': 1,
                                 'Tier_2': 2,
                                 'Low': 0, 'Medium': 1, 'High': 2, 'None': None}, inplace=False)

                df = df.dropna(how='any')

                self.datalist = df
                self.labels = cols
                self.X = torch.from_numpy(np.array(df[cols[1:]], dtype=np.float16))
                self.y = torch.from_numpy(np.array(df['Dependent-Company Status'],
                                          dtype=np.float16))

        def __getitem__(self, index):
                return self.X[index], self.y[index]

        def __len__(self):
                return len(self.X)

class MLP(nn.Module):
        def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                        nn.Linear(39, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                )

        def forward(self, x):
                return self.layers(x)


if __name__ == '__main__':
        torch.manual_seed(42)
        trainloader = torch.utils.data.DataLoader(__Dataset(), batch_size=10, shuffle=True)
        mlp = MLP()
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

        for epoch in range(0, 5):
                print('Starting epoch: ', epoch)
                current_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                        inputs, targets = data
                        inputs, targets = inputs.float(), targets.float()
                        targets = targets.reshape((targets.shape[0], 1))

                        optimizer.zero_grad()

                        outputs = mlp(inputs)

                        print(outputs)

                        loss = loss_function(outputs, targets)
                        loss.backward()

                        optimizer.step()

                        # current_loss += loss
                        # if i % 10 == 0:
                        #         print('Loss after mini-batch %5d: ' %
                        #               (i + 1, current_loss))
                        #         print(current_loss)
                        #         current_loss = 0.0

        print('Training process has finished.')