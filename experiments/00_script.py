import os 
import numpy as np

# lrs = [1.0, 2.0, 5.0]
# betas = [0.0, 0.9]
betas = [0.8, 0.85, 0.95]
epochs = 200
lmd = 0.0
optimizer = 'STP'
run_name = "exp_mushrooms.py"
for a in betas:
    os.system(f"python {run_name} --n-epochs={epochs} --optimizer={optimizer} --lmd={lmd} --seed=-1 --beta={a}")