import random

# Monty Hall problem
# Simulation of the two strategies gains

random.seed(0)

R = 20000
win = 0
win2 = 0

for r in range(1, R + 1):
    Car = random.randint(1, 3)
    Pick = random.randint(1, 3)
    
    Open = random.choice([i for i in range(1, 4) if i not in {Pick, Car}])
    
    if sum(1 for i in range(1, 4) if i not in {Pick, Car}) == 1:
        Open = next(i for i in range(1, 4) if i not in {Pick, Car})
    
    Pick2 = random.choice([i for i in range(1, 4) if i not in {Pick, Open}])
    
    if sum(1 for i in range(1, 4) if i not in {Pick, Open}) == 1:
        Pick2 = next(i for i in range(1, 4) if i not in {Pick, Open})
    
    if Pick == Car:
        win += 1
    
    if Pick2 == Car:
        win2 += 1
    
    if Pick == Pick2:
        raise Exception("Error")
    
    print(f"Car door= {Car} Pick= {Pick} Open= {Open} Pick2= {Pick2} : Change: % wins= {win2/r:.4f} No Change: % wins= {win/r:.4f}")
    # input("Press [enter] to continue")

