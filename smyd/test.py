import pandas as pd
import matplotlib.pyplot as plt
file_path ='smyd/hv_roundabout_v1_dense.csv'
data = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.plot(data['X'], data['Y'], marker='o', linestyle='-')
plt.show()