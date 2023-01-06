import numpy as np
import pandas as pd
import os
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
from tqdm import tqdm


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
                                 'Masters': 2, 'PhD': 3, 'None': 0, 'Both': 3, 'Tier_1': 1,
                                 'Tier_2': 2,
                                 'Low': 0, 'Medium': 1, 'High': 2}, inplace=False)

                df = df.dropna(how='any')

                self.datalist = df
                self.labels = cols
                # x = np.array(df[cols[1:]], dtype=np.float16)
                # y = df['Dependent-Company Status']

        def __getitem__(self, index):
                return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]

        def __len__(self):
                return self.datalist.shape[0]

        def __size__(self):
                return self.datalist.shape


train_set = __Dataset()
trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=40, shuffle=False)

print(train_set.__size__())


class Net(nn.Module):
        def __init__(self):
                super().__init__()

                self.fc1 = nn.Linear(20, 40)
                self.b1 = nn.BatchNorm1d(40)
                self.fc2 = nn.Linear(40, 20)
                self.b2 = nn.BatchNorm1d(20)
                self.fc3 = nn.Linear(20, 10)
                self.b3 = nn.BatchNorm1d(10)
                self.fc4 = nn.Linear(4, 1)

        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.b1(x)
                x = F.relu(self.fc2(x))
                x = self.b2(x)
                x = F.relu(self.fc3(x))
                x = self.b3(x)
                x = F.sigmoid(self.fc4(x))

                return x

from torch.optim import Adam

net = Net()
criterion = nn.MSELoss()
EPOCHS = 200
optm = Adam(net.parameters(), lr=0.001)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(model, x, y, optimizer, criterion):
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        return loss, output


for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        for bidx, batch in enumerate(trainloader):
                x_train, y_train = batch['inp'], batch['oup']
                x_train = x_train.view(-1, 8)
                x_train = x_train.to(device)
                y_train - y_train.to(device)
                loss, predictions = train(net, x_train, y_train, optm, criterion)
                for idx, i in enumerate(predictions):
                        i = torch.round(i)
                        if i == y_train[idx]:
                                correct += 1
                acc = (correct/len(data))
                epoch_loss += loss

        print('Epoch {} Accuracy: {}'.format(epoch+1, acc*100))
        print('Epoch {} Loss: {}'.format((epoch+1), epoch_loss))
