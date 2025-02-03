## # "Statistical foundations of machine learning" software
# utils.py
# Author: G. Bontempi
# mcarlo4.py
# Library of regression learning algorithms

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from  mrmr import mrmr_regression

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from annoy import AnnoyIndex

def find_knn(data, query_point, k, metric='euclidean'):
  """Finds the K nearest neighbors of a query point in a dataset.

  Args:
    data: A NumPy array representing the dataset.
    query_point: A NumPy array representing the query point.
    k: The number of nearest neighbors to find.
    metric: The distance metric to use ('euclidean', 'angular', etc.).

  Returns:
    A tuple containing two NumPy arrays:
      - The indices of the K nearest neighbors in the dataset.
      - The distances to the K nearest neighbors.
  """

  num_dimensions = data.shape[1]
  index = AnnoyIndex(num_dimensions, metric)

  for i, point in enumerate(data):
    index.add_item(i, point)

  index.build(10)  # Build the index with 10 trees (adjust as needed)

  indices = index.get_nns_by_vector(query_point, k, include_distances=False)

  return indices

## Methods
## lazy_regr0
## lazy_regr
## keras0_regr
## pls_regr
## ridge_regr
## lasso_regr
## enet_egr
## rf_regr
## rf_regr0
## lin_regr
## knn_regr
## gb_regr
## ab_regr
## piperf_regr
## pipeknn_regr
## pipelin_regr
## pipeab_regr
## torch_regr"
  
def predpy(algo, X_train, y_train, X_test,params={ "nepochs":100}):
    m=y_train.shape[1]
    nepochs=100
    yhat=0
    hidden=10
    

    if "nepochs" in params:
        nepochs=params["nepochs"]
        
    if "hidden" in params:
        hidden=params["hidden"]

    Nts=X_test.shape[0]
    n=X_train.shape[1]
    if algo=="lazy_regr0":

        k = 5  # Find the 5 nearest neighbors
        yhat=np.zeros(Nts)
        for i in np.arange(Nts):
            indices = find_knn(X_train, X_test[i,:], k)
            yhat[i]=np.mean(y_train[indices])
            
    if algo=="lazy_regr":
        from sklearn.linear_model import LassoCV
        k = 3*n  
        yhat=np.zeros(Nts)
        selected_features = mrmr_regression(pd.DataFrame(X_train), y_train, K=5)
        X_train=X_train[:,selected_features]
        X_test=X_test[:,selected_features]
        for i in np.arange(Nts):
            indices = find_knn(X_train, X_test[i,:], k)
            Xl=X_train[indices,:]
            Yl=y_train[indices]
            reg = LassoCV(cv=2, random_state=0).fit(Xl, Yl)
        
            yhat[i] = reg.predict(X_test[i,:].reshape(1,-1))
            
    
    if algo=="keras0_regr":
        from tensorflow import keras 
        from tensorflow.keras import layers
        def build_model():
          model = keras.Sequential([              
              layers.Dense(4, activation="relu"),
              layers.Dropout(0.95),
              layers.Dense(2, activation="relu"),
              layers.Dense(m)
          ])
          model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
          return model
        model = build_model()                                
        model.fit(X_train, y_train,                 
                epochs=nepochs, batch_size=10, verbose=0, validation_split=0.2)

        yhat=model.predict(X_test)
    
    if algo=="keras_regr":
    # Based on Keras Tuner
    ## https://keras.io/keras_tuner/
    ##  https://www.tensorflow.org/tutorials/keras/keras_tuner
        from tensorflow import keras 
        from tensorflow.keras import layers
        import keras_tuner as kt
        import tensorflow as tf
        def model_builder(hp):
              model = keras.Sequential()
              # Tune the number of units in the first Dense layer
              # Choose an optimal value between 32-512
              hp_units = hp.Int('units', min_value=1, max_value=20, step=1)
              model.add(keras.layers.Dense(units=hp_units, activation='relu'))
              hp_units2 = hp.Int('units2', min_value=2, max_value=10, step=2)
              model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
              hp_droprate = hp.Choice('droprate', values=[0.1, 0.5, 0.7, 0.9])
              model.add(keras.layers.Dropout(hp_droprate))
              model.add(keras.layers.Dense(m))

              model.compile(optimizer="rmsprop",
                          loss="mse",
                          metrics=['accuracy'])

              return model

        tuner = kt.Hyperband(model_builder,
                           objective='val_accuracy',
                           max_epochs=10,
                           factor=3)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(X_train, y_train,
          epochs=50, validation_split=0.2, callbacks=[stop_early],verbose=0)

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history =model.fit(X_train, y_train,epochs=nepochs, batch_size=10, 
          validation_split=0.25, verbose=0)

        #val_acc_per_epoch = history.history['val_accuracy']
        #best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1                              
        #hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        #hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2,verbose=0)
        yhat=model.predict(X_test)
    
    
    if algo=="pls_regr":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import RandomizedSearchCV
        random_grid = {'n_components': [int(x) for x in np.linspace(1, 20, num = 15)]}
        reg = PLSRegression()

        pls_regressor = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,
        n_iter = 70, cv = 3, verbose=0, random_state=42)

        pls_regressor.fit(X_train, y_train)
        yhat = pls_regressor.predict(X_test)
    
    if algo=="ridge_regr":
        from sklearn.linear_model import RidgeCV

        reg = RidgeCV(alphas=np.linspace(0.1, 100, num = 50))
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)

    
    if algo=="lasso_regr":  
        from sklearn.linear_model import LassoCV, MultiTaskLassoCV
        if m==1:
          reg = LassoCV(cv=10, random_state=0).fit(X_train, y_train)
        else:
          reg = MultiTaskLassoCV(cv=2, random_state=0,max_iter=10,
          verbose=0).fit(X_train, y_train)
        yhat = reg.predict(X_test)
    
    if algo=="enet_regr":  
        from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
        from sklearn.model_selection import RandomizedSearchCV

        if m==1:
          reg = ElasticNet(random_state=0).fit(X_train, y_train)
        else:
          random_grid = {'alpha': [int(x) for x in np.linspace(0.1, 2, num = 10)]}
          reg =MultiTaskElasticNet()
          reg = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,
          n_iter = 50, cv = 3, verbose=0, random_state=42)
          reg.fit(X_train, y_train)

        yhat = reg.predict(X_test)
    
    
    
    if algo=="lin_regr":
        # Create linear regression object
        from sklearn import linear_model
        linear_regressor = linear_model.LinearRegression()
        linear_regressor.fit(X_train, y_train)
        yhat = linear_regressor.predict(X_test)
  
    if algo=="rf_regr0":
        from sklearn.ensemble import RandomForestRegressor
        #from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV

        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

        # Create the random grid
        random_grid = {'max_depth': max_depth}
        #             'min_samples_split': min_samples_split,
        #               'min_samples_leaf': min_samples_leaf}
        rf_r = RandomForestRegressor()
        rf_regressor = RandomizedSearchCV(estimator = rf_r, param_distributions = random_grid,
          n_iter = 5, cv = 3, verbose=0, random_state=42)
        rf_regressor =rf_r
        rf_regressor.fit(X_train, y_train)
        yhat = rf_regressor.predict(X_test)
  
    if algo=="rf_regr":
        from sklearn.ensemble import RandomForestRegressor
        #from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        rf_r = RandomForestRegressor()
        if m>1:
            from sklearn.multioutput import RegressorChain
            rf_r = RegressorChain(base_estimator=rf_r, order='random')
        
        rf_regressor =rf_r
        rf_regressor.fit(X_train, y_train)

        yhat = rf_regressor.predict(X_test)
    
   
    if algo=="knn_regr":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import RandomizedSearchCV

        # Create the random grid
        random_grid = {'n_neighbors': [int(x) for x in np.linspace(3, 20, num = 10)],
                  'weights':['uniform', 'distance']}
        knn_r = KNeighborsRegressor()

        knn_regressor = RandomizedSearchCV(estimator = knn_r, param_distributions = random_grid,
        n_iter = 50, cv = 3, verbose=0, random_state=42)
        knn_regressor.fit(X_train, y_train)
        yhat = knn_regressor.predict(X_test)
    
    if algo=="gb_regr":
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        if m>1:
          #from sklearn.multioutput import MultiOutputRegressor
          #gb_regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
          from sklearn.multioutput import RegressorChain
          gb_regressor = RegressorChain(base_estimator=GradientBoostingRegressor(), order='random')
          random_grid = {'base_estimator__n_estimators': [int(x) for x in np.linspace(1, 20, num = 5)]}
        else:
          gb_regressor = GradientBoostingRegressor()
          random_grid = {'n_estimators': [int(x) for x in np.linspace(1, 20, num = 5)]}

        gb_regressor = RandomizedSearchCV(estimator = gb_regressor, param_distributions = random_grid,
        n_iter = 20, cv = 2, verbose=0, random_state=42)

        gb_regressor.fit(X_train, y_train)

        yhat = gb_regressor.predict(X_test)
   
    
    if algo=="ab_regr":
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        if m>1:
          from sklearn.multioutput import MultiOutputRegressor
          ab_regressor = MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), 
            n_estimators=400, random_state=7))
          random_grid = {'estimator__base_estimator__max_depth': [int(x) for x in np.linspace(1, 10, num = 5)]}
        else:
          ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
          random_grid = {'base_estimator__max_depth': [int(x) for x in np.linspace(1, 10, num = 5)]}

        ab_regressor = RandomizedSearchCV(estimator = ab_regressor, param_distributions = random_grid,
        n_iter = 20, cv = 2, verbose=0, random_state=42)

        ab_regressor.fit(X_train, y_train)
        yhat = ab_regressor.predict(X_test)
    
  
  
    
    if algo=="piperf_regr":  
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import RegressorChain
        #if m>1:
        #  clf = Pipeline([
        #  ('feature_selection', SelectFromModel(RandomForestRegressor())),
        #  ('regression', RegressorChain(base_estimator=RandomForestRegressor(), order='random'))
        #  ])
        #else:
        clf = Pipeline([
          ('feature_selection', SelectFromModel(RandomForestRegressor())),
          ('regression', RandomForestRegressor())
          ])
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
    
    if algo=="pipeknn_regr":  
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectFromModel
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        knn_r = Pipeline([
          ('feature_selection', SelectFromModel(RandomForestRegressor())),
          ('regression', KNeighborsRegressor())
        ])
        random_grid = {'feature_selection__max_features': [int(x) for x in np.linspace(1, 10, num = 5)],
        'regression__n_neighbors': [int(x) for x in np.linspace(1, 20, num = 5)]}
        knn_regressor = RandomizedSearchCV(estimator = knn_r, param_distributions = random_grid,
        n_iter = 20, cv = 2, verbose=0, random_state=42)
        knn_regressor.fit(X_train, y_train)
        yhat = knn_regressor.predict(X_test)
    
    if algo=="pipelin_regr":  
        clf = Pipeline([
          ('reduce_dim', PCA()),
          ('feature_selection', SelectFromModel(RandomForestRegressor())),
          ('regression', linear_model.LinearRegression())
        ])
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)

    
    if algo=="pipeab_regr":  
        clf = Pipeline([
          ('feature_selection', SelectFromModel(RandomForestRegressor())),
          ('regression', AdaBoostRegressor(n_estimators=500))
        ])
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)

        yhat.shape=(int(Nts),int(m))
    
    if algo=="torch_regr":

        def torchtrain(Xtr, Ytr, n_epochs, optimizer, model, loss_fn):
            for epoch in range(1, n_epochs + 1):
                Yhatr = model(Xtr) 
                loss_train = loss_fn(Yhatr, Ytr)
                optimizer.zero_grad()
                loss_train.backward() # <2>
                optimizer.step()
                
        seq_model = nn.Sequential(
            nn.Linear(n, hidden), # <1>
            nn.Tanh(),
            nn.Linear(hidden, m)) # <2>
        
        optimizer = optim.Adam(seq_model.parameters(), lr=1e-2)
        params=torchtrain(
            n_epochs = epochs, 
            optimizer = optimizer,
            model = seq_model,
            loss_fn = nn.MSELoss(),
            Xtr = X_trainT,
            Ytr = y_trainT)

        yhat=seq_model(X_testT).detach().numpy()
        yhat.reshape(Nts,m)  

    if algo=="torch2_regr":

        # Convert to PyTorch tensors
        X_trainT = torch.tensor(X_train, dtype=torch.float32)
        y_trainT = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Unsqueeze for shape
        X_testT = torch.tensor(X_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_trainT, y_trainT)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_testT)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. Define the Neural Network Model:
        class RegressionModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(RegressionModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc1a = nn.Linear(hidden_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
              out = self.fc1(x)
              out = self.relu(out)
              #out = self.fc1a(out)
              #out = self.relu(out)
              out = self.fc2(out)
              return out

        # Instantiate the model
        input_size = X_trainT.shape[1]  # Number of features
        hidden_size = hidden  # Adjust as needed
        output_size = m   # For regression
        model = RegressionModel(input_size, hidden_size, output_size)


        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()  # Mean Squared Error for regression

      # Training loop
        epochs = nepochs  # Adjust as needed
    
        for epoch in range(epochs):
          for batch_idx, (data, target) in enumerate(train_loader):
              optimizer.zero_grad()
              output = model(data)
              loss = criterion(output, target)
              loss.backward()
              optimizer.step()
          #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        model.eval()  # Set the model to evaluation mode
        yhat=[]
        #import pdb
        with torch.no_grad():  # Don't calculate gradients during evaluation
          total_loss = 0
          for data in test_loader:
              #pdb.set_trace()  
              output = model(data[0])
              yhat=np.append(yhat,output.numpy())
          


    if type(yhat)==int:
        print(algo)
        raise ValueError("Method not present")
    return yhat  


############
## sgd_class
## nb_class
## knn_class
## rf_class
## ab_class
## svm_class
## lsvm_class
## gp_class
## gb_class
## piperf_class


def classpy(algo, X_train, y_train, X_test,params={"m":1, "nepochs":1000}):
    m=1
    nepochs=100
    yhat=0
    hidden=20
    
    if "m" in params:
        m=params["m"]

    if "nepochs" in params:
        nepochs=params["nepochs"]
        
    if "hidden" in params:
        hidden=params["hidden"]

    Nts=X_test.shape[0]
    Ntr,n=X_train.shape
  
    yhat=[]

    if algo=="lazy_class":
        from sklearn.linear_model import LassoCV
        k = 3*n  
        yhat=np.zeros(Nts)
        phat=np.zeros((Nts,2))
        selected_features = mrmr_regression(pd.DataFrame(X_train), y_train, K=5)
        X_train=X_train[:,selected_features]
        X_test=X_test[:,selected_features]
        for i in np.arange(Nts):
            indices = find_knn(X_train, X_test[i,:], k)
            Xl=X_train[indices,:]
            Yl=y_train[indices]
            reg = LassoCV(cv=5, random_state=0).fit(Xl, Yl)
            phat[i,1] = min(max(0,reg.predict(X_test[i,:].reshape(1,-1))),1)
            phat[i,0] =1-phat[i,1] 
            yhat[i]=phat[i,1]>0.5
            
    if algo=="sgd_class": 
      from sklearn.linear_model import SGDClassifier
      sgd_clf = SGDClassifier(loss='log_loss',random_state=42)
      sgd_clf.fit(X_train, y_train)
      yhat = sgd_clf.predict(X_test)
      phat = sgd_clf.predict_proba(X_test) 
 

  
    if algo=="nb_class":
      from sklearn.naive_bayes import GaussianNB
      gnb = GaussianNB()
      gnb.fit(X_train, y_train)
      yhat = gnb.predict(X_test) 
      phat = gnb.predict_proba(X_test)   
  
    if algo=="knn_class":
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.model_selection import RandomizedSearchCV
    
      # Create the random grid
      random_grid = {'n_neighbors': [int(x) for x in np.linspace(1, 20, num = 10)],
                    'weights':['uniform', 'distance']}
      neigh_r = KNeighborsClassifier(n_neighbors=3)
      neigh = RandomizedSearchCV(estimator = neigh_r, param_distributions = random_grid,
      n_iter = 50, cv = 3, verbose=0, random_state=42)
      neigh.fit(X_train, y_train)
      yhat = neigh.predict(X_test) 
      phat = neigh.predict_proba(X_test) 
  
    if algo=="rf_class":
      from sklearn.ensemble import RandomForestClassifier
      clf = RandomForestClassifier(max_depth=2, random_state=0)
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test) 
      phat = clf.predict_proba(X_test) 
      
    if algo=="ab_class":
      from sklearn.ensemble import AdaBoostClassifier
      clf = AdaBoostClassifier(n_estimators=100, random_state=0)
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test) 
      phat = clf.predict_proba(X_test) 
      
    if algo=="svm_class":
      from sklearn.svm import SVC
      clf = SVC(gamma='auto',probability=True)
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test) 
      phat = clf.predict_proba(X_test) 
      
    if algo=="lsvm_class":
      from sklearn.svm import SVC
      clf = SVC(kernel="linear", C=0.025,probability=True)
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test) 
      phat = clf.predict_proba(X_test)
  
    if algo=="gp_class":
      from sklearn.gaussian_process import GaussianProcessClassifier
      from sklearn.gaussian_process.kernels import RBF
      kernel = 1.0 * RBF(1.0)
      gpc = GaussianProcessClassifier(kernel=kernel,
           random_state=0)
      gpc.fit(X_train, y_train)
      yhat = gpc.predict(X_test) 
      phat = gpc.predict_proba(X_test)
      
    if algo=="gb_class":
      from sklearn.ensemble import GradientBoostingClassifier
      gb_classifier = GradientBoostingClassifier()
      gb_classifier.fit(X_train, y_train)
      yhat = gb_classifier.predict(X_test)
      phat = gb_classifier.predict_proba(X_test) 
      
    if algo=="piperf_class":  
      from sklearn.pipeline import Pipeline
      from sklearn.feature_selection import SelectFromModel
      from sklearn.ensemble import RandomForestClassifier
      clf = Pipeline([
        ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ('regression', RandomForestClassifier())
      ])
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test)
      phat = clf.predict_proba(X_test) 
  
    
    if algo=="pipeknn_class":  
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        clf = Pipeline([
            ('feature_selection', SelectFromModel(RandomForestClassifier())),
            ('classification', KNeighborsClassifier(n_neighbors=10))
            ])
        
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
        phat = clf.predict_proba(X_test) 
        
    if algo=="torchlogit_class": 
        import torch   
        import torch.nn as nn
        def training_class(Xtr, Ytr, n_epochs, optimizer, model, loss_fn):
            for epoch in range(1, n_epochs + 1):
                
                    Yhatr = model(Xtr) # <1>
                    
                    loss_train = loss_fn(Yhatr.reshape(Ntr,m), Ytr.reshape(Ntr,m))
        
                    
                    optimizer.zero_grad()
                    loss_train.backward() # <2>
                    optimizer.step()
            
                
        tensorX = torch.from_numpy(X_train).type(torch.float)
        tensorY = torch.from_numpy(y_train).type(torch.float)
        tensorXts = torch.from_numpy(X_test).type(torch.float)
        
        seq_model = nn.Sequential(
            nn.Linear(n, hidden), 
            nn.Tanh(),
            nn.Linear(hidden,1)) 
        optimizer = optim.Adam(seq_model.parameters(), lr=1e-2)
        params=training_class(
            n_epochs = nepochs, 
            optimizer = optimizer,
            model = seq_model,
            loss_fn = nn.BCEWithLogitsLoss(),
            Xtr = tensorX,
            Ytr = tensorY)
        phat=torch.sigmoid(seq_model(tensorXts)).detach().numpy().reshape(Nts,1)
        yhat = np.round(phat).reshape(Nts,1)
        phat=1-phat
    
    if algo=="torchcross_class": 
        import torch   
        import torch.nn as nn
        tensorX = torch.from_numpy(X_train).type(torch.float)
        tensorY = torch.from_numpy(y_train).type(torch.LongTensor)
        tensorXts = torch.from_numpy(X_test).type(torch.float)
        train_dataset = TensorDataset(tensorX, tensorY )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                                   shuffle=True)

        model = nn.Sequential(
                    nn.Linear(n, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 2))

        learning_rate = 1e-2
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        loss_fn = nn.CrossEntropyLoss()



        for epoch in range(nepochs):
            for imgs, labels in train_loader:
                outputs = model(imgs)
                
                loss = loss_fn(outputs, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            
                

        pred=model(tensorXts)
        phat=torch.exp(pred).detach().numpy()
        _, Yhats = torch.max(pred,dim=1)
        yhat=Yhats.detach().numpy().reshape(Nts,m)
        
        
    if algo=="torchBCE_class": 
        ## use of DataLoader
            import torch   
            import torch.nn as nn
            tensorX = torch.from_numpy(X_train).type(torch.float)
            tensorY = torch.from_numpy(y_train).type(torch.float)
            tensorXts = torch.from_numpy(X_test).type(torch.float)
            train_dataset = TensorDataset(tensorX, tensorY )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                                       shuffle=True)

            model = nn.Sequential(
                        nn.Linear(n, hidden),
                        nn.Tanh(),
                        nn.Linear(hidden, 1))


            optimizer = optim.SGD(model.parameters(), lr=1e-2)
            
            loss_fn = nn.BCEWithLogitsLoss() 
            

            for epoch in range(nepochs):
                for imgs, labels in train_loader:
                    Nl=imgs.shape[0]
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels.reshape(Nl,1))
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            
            pred=model(tensorXts)
            phat=1-torch.sigmoid(pred).detach().numpy()
            yhat=torch.sigmoid(pred).round().detach().numpy().reshape(Nts,m)



    if algo=="torchMLP_class": 
        ## use of DataLoader
            import torch   
            import torch.nn as nn
            tensorX = torch.from_numpy(X_train).type(torch.float)
            tensorY = torch.from_numpy(y_train).type(torch.float)
            tensorXts = torch.from_numpy(X_test).type(torch.float)
            train_dataset = TensorDataset(tensorX, tensorY )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                                       shuffle=True)

            model = nn.Sequential(
                        nn.Linear(n, hidden),
                        nn.Tanh(),
                        nn.Linear(hidden, 1),
                        nn.Sigmoid())


            optimizer = optim.SGD(model.parameters(), lr=1e-2)
            
            loss_fn = nn.BCEWithLogitsLoss() 
            

            for epoch in range(nepochs):
                for imgs, labels in train_loader:
                    Nl=imgs.shape[0]
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels.reshape(Nl,1))
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            
            pred=model(tensorXts)
            phat=1-pred.detach().numpy()
            yhat=pred.round().detach().numpy().reshape(Nts,m)


    if algo=="torchSoft_class": 
      ## use of DataLoader
          import torch   
          import torch.nn as nn
          tensorX = torch.from_numpy(X_train).type(torch.float)
          tensorY = torch.from_numpy(y_train).type(torch.float)
          tensorXts = torch.from_numpy(X_test).type(torch.float)
          train_dataset = TensorDataset(tensorX, tensorY )
          train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                                     shuffle=True)

          model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh())


          optimizer = optim.Adam(model.parameters(), lr=1e-3)
          
          loss_fn = nn.SoftMarginLoss()
          

          for epoch in range(nepochs):
              for imgs, labels in train_loader:
                  Nl=imgs.shape[0]
                  labels[labels == 0] = -1
                  outputs = model(imgs)
                  loss = loss_fn(outputs, labels.reshape(Nl,1))
          
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

          
          pred=(model(tensorXts)+1)/2
          phat=1-pred.detach().numpy()
          yhat=pred.round().detach().numpy().reshape(Nts,m)



## old version
    if type(yhat)==int:
        print(algo)
        raise ValueError("Method not present")
    return yhat,phat  


def Embed(tseries,lag=2,H=1):
    m=tseries.shape[1]
    X=None
    
    for j in np.arange(m):
        Embed = np.lib.stride_tricks.sliding_window_view(tseries[:,j], window_shape=lag+H)
        if X is None:
            X=Embed[:,:lag]
            Y=Embed[:,-H:]
        else:
            X = np.column_stack((X, Embed[:,:lag]))
            Y=np.column_stack((Y, Embed[:,-H:]))
    
    
    return X,Y



## old version
def EmbedTS(TS,n=2,H=1):
    N,m=TS.shape
    for i in np.arange(N - n - H - 1):
        Xi = TS[i:(i + n), 0].reshape(1, -1)
        Yi = TS[(n + i):(i + n + H), 0].reshape(1, -1)
        for j in np.arange(1, m):
            Xi = np.hstack((Xi, TS[i:(i + n), j].reshape(1, -1)))
            Yi = np.hstack((Yi, TS[(n + i):(i + n + H), j].reshape(1, -1)))
        if i == 0:
            X = Xi
            Y = Yi
        else:
            X = np.vstack((X, Xi.reshape(1, -1)))
            Y = np.vstack((Y, Yi.reshape(1, -1)))
    return X,Y