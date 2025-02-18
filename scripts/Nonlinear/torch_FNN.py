#!/usr/bin/env python3
## # "Statistical foundations of machine learning" software
# torch_FNN.py
# Author: G. Bontempi
## Implementation of a FNN using pytorch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


## FNN with 2 hidden layers
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs,H1=10,H2=10):
        super().__init__()

        self.layers = torch.nn.Sequential(
            
            # 1st hidden layer
            torch.nn.Linear(num_inputs, H1),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(H2, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

n=50 ## number of inputs
m=3 ## number of outputs

model = NeuralNetwork(n, m)
print(model)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

torch.manual_seed(123)


print(model.layers[0].weight)

print(model.layers[0].weight.shape)






def f(x,sdw=0.5,m=1):
    N=x.shape[0]
    y=torch.abs(x[:,0]*x[:,2]*x[:,3])+torch.pow(x[:,1],2.0)+torch.log(torch.abs(x[:,-1]))
    y=y.reshape(N,1)+sdw*torch.randn((N, m))
    return(y)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N=100
m=1
Nts=1000
sdw=0.2
X_train = torch.rand((N, n))
Y_train = f(X_train,sdw=sdw)
X_test = torch.rand((Nts, n))
Y_test = f(X_test,sdw=sdw)
model = NeuralNetwork(n, m,10,10)



class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]        
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, Y_train)
test_ds = ToyDataset(X_test, Y_test)


train_loader = DataLoader(
    dataset=train_ds,
    batch_size=5,
    shuffle=True,
    num_workers=0
)


test_loader = DataLoader(
    dataset=test_ds,
    batch_size=5,
    shuffle=False,
    num_workers=0
)



torch.manual_seed(123)
model = NeuralNetwork(num_inputs=n, num_outputs=m)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

num_epochs = 30

for epoch in range(num_epochs):
    
    #model.train()
    for batch_idx, (x, y) in enumerate(train_loader):

        yhat = model(x)
        
        loss = F.mse_loss(yhat, y) # Loss function
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        ### LOGGING
        if epoch%10==0:
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    # Optional model evaluation
    
with torch.no_grad():
    Yhat = model(X_train)
    Yhats = model(X_test)


print('\n --- \n MSEtrain=',F.mse_loss(Yhat, Y_train) ,
      '\n MSEtest=',F.mse_loss(Yhats, Y_test) )