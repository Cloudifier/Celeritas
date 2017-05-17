
import pandas as pd
import numpy as np
import json
import time as tm

import os
from sql_helper import MSSQLHelper
from datetime import datetime as dt

import matplotlib.pyplot as plt

np.set_printoptions(precision = 3, suppress = True)

sqleng = MSSQLHelper()

df = sqleng.Select("SELECT [ClusterId],[R],[M] FROM [dbo].[z2016rfm4]")
df = df.sample(frac=1)
print("Plotting {} rows dataset".format(df.shape[0]))
plt.figure(figsize=(15,15))
t0=tm.time()
plt.scatter(df.R, df.M, c = df.ClusterId)
plt.show()
t1=tm.time()
print("Plotting time {:.2f}s".format(t1-t0))