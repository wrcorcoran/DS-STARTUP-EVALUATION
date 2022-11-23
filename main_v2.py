import torch
import os
import numpy as np
import pandas as pd
from torch.nn import Linear, Sigmoid, MSELoss
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch import nn, optim
import torch.nn.functional as F

MODEL_PATH = 'model.pth'

def calculate_accuracy(y_true, y_pred):
        predicted = y_pred.ge(.5).view(-1)
        return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
        return round(t.item(), decimal_places)


class MLP(nn.Module):
        def __init__(self, n_inputs):
                super(MLP, self).__init__()
                self.layer = Linear(n_inputs, 1)
                self.activation = Sigmoid()

        def forward(self, X):
                X = self.layer(X)
                X = self.activation(X)
                return X

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

x = np.array(df[cols[1:]], dtype=np.float16)
y = df['Dependent-Company Status']

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
#
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
#
#
# if os.path.isfile(MODEL_PATH):
#         net = torch.load(MODEL_PATH)
# else:
#         net = Net(X_train.shape[2])
#
#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# X_train = X_train.to(device)
# y_train = y_train.to(device)
#
# X_test = X_test.to(device)
# y_test = y_test.to(device)
#
# net = net.to(device)
#
# criterion = criterion.to(device)
#
# for epoch in range(1000):
#         y_pred = net(X_train)
#
#         y_pred = torch.squeeze(y_pred)
#         train_loss = criterion(y_pred, y_train)
#
#         if epoch % 100 == 0:
#                 train_acc = calculate_accuracy(y_train, y_pred)
#
#                 y_test_pred = net(X_test)
#                 y_test_pred = torch.squeeze(y_test_pred)
#
#                 test_loss = criterion(y_test_pred, y_test)
#
#                 test_acc = calculate_accuracy(y_test, y_test_pred)
#                 print(
#                         f'''epoch {epoch}
#                         Train set - loss: {round_tensor(train_loss)}, accuracy:{round_tensor(train_acc)}
#                         Test set - loss: {round_tensor(test_loss)}, accuracy:{round_tensor(test_acc)}''')
#
#                 optimizer.zero_grad()
#                 train_loss.backward()
#                 optimizer.step()
#
# torch.save(net, MODEL_PATH)
#
# classes = ['Failed', 'Success']
#
# y_pred = net(X_test)
#
# y_pred = y_pred.ge(.5).view(-1).cpu()
# y_test = y_test.cpu()
#
# print(classification_report(y_test, y_pred, target_names=classes))

train_set_size = int(len(df) * 0.8)
valid_set_size = len(df) - train_set_size
train, test = random_split(df, [train_set_size, valid_set_size])

train_dl = DataLoader(train, batch_size = 32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)

criterion = MSELoss()
optimizer = SGD(model.paramaters(), lr=0.01, momentum=0.9)

