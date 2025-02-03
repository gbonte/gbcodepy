# "Statistical foundations of machine learning" software
# Python module gbcode 
# Author: G. Bontempi
# pizza.py
# Example on conditional probability from the Handbook
import pandas as pd

def P(model, **kwargs):
    M = model.copy()
    for key, value in kwargs.items():
        M = M[M[key] == value]
    return M['p'].sum()

# Define the model DataFrame
Model = pd.DataFrame({
    'owner': ["IT","BE","IT","BE","IT","BE","IT","BE"],
    'cook': ["IT","IT","BE","BE","IT","IT","BE","BE"],
    'pizza': ["GOOD","GOOD","GOOD","GOOD","BAD","BAD","BAD","BAD"],
    'p': [0.378, 0.168, 0.012, 0.032, 0.162, 0.072, 0.048, 0.128]
})

print("P(pizza=GOOD)", P(Model, pizza="GOOD"))
print("P(pizza=GOOD| owner=italian)", P(Model, pizza="GOOD", owner="IT") / P(Model, owner="IT"))
print("P(pizza=GOOD| owner=belgian)",P(Model, pizza="GOOD", owner="BE") / P(Model, owner="BE"))

# P(pizza|owner="italian", cook="italian") = P(pizza|cook="italian") = P(pizza|owner="belgian", cook="italian")
print("(pizza|owner=italian, cook=italian)=", P(Model, pizza="GOOD", cook="IT", owner="IT") / P(Model, cook="IT", owner="IT"))
print("P(pizza|cook=italian)=", P(Model, pizza="GOOD", cook="IT") / P(Model, cook="IT"))
print("P(pizza|owner=belgian, cook=italian)", P(Model, pizza="GOOD", cook="IT", owner="BE") / P(Model, cook="IT", owner="BE"))

# P(pizza|owner="italian", cook="italian") = P(pizza|cook="italian") = P(pizza|owner="belgian", cook="italian")
print("(pizza|owner=italian, cook=italian)", P(Model, pizza="GOOD", cook="BE", owner="IT") / P(Model, cook="BE", owner="IT"))
print("P(pizza|cook=italian) ", P(Model, pizza="GOOD", cook="BE") / P(Model, cook="BE"))
print("P(pizza|owner=belgian, cook=italian)", P(Model, pizza="GOOD", cook="BE", owner="BE") / P(Model, cook="BE", owner="BE"))
