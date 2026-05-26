import torch.nn as nn

class MoistExchangesNN(nn.Module):
    def __init__(self):
        super(MoistExchangesNN, self).__init__()
        self.fc1 = nn.Linear(5,20)
        self.fc2 = nn.Linear(20,80)
        self.fc3 = nn.Linear(80,80)
        self.fc4 = nn.Linear(80,20)
        self.fc5 = nn.Linear(20,4)
        self.fc6 = nn.Linear(4,1)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.relu5 = nn.LeakyReLU()

        # set some initial values
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x

class MoistExchangesNN_vli(nn.Module):
    def __init__(self):
        super(MoistExchangesNN, self).__init__()
        self.fc1 = nn.Linear(7,24)
        self.fc2 = nn.Linear(24,96)
        self.fc3 = nn.Linear(96,96)
        self.fc4 = nn.Linear(96,96)
        self.fc5 = nn.Linear(96,24)
        self.fc6 = nn.Linear(24,3)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.relu5 = nn.LeakyReLU()

        # set some initial values
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x

