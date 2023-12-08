import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['Yeast','Scene','Flags','Emotions','Amazon','Mirflickr']
original = [1.88375,0.12165,0.7647748,0.179571,1.5224314,1.24397]
MLGAN = [0.6929226,0.15698,0.4133116,0.1623664,1.0592099,0.6736]
MLSMOTE=[0.747221,0.151388,0.56302,0.202864,1.5826495,0.7]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.09, original, 0.175, label = 'Original')
plt.bar(X_axis + 0.1, MLGAN, 0.175, label = 'MLGAN')
plt.bar(X_axis + 0.29, MLSMOTE, 0.175, label = 'MLSMOTE')
  
plt.xticks(X_axis, X)
plt.xlabel("Datasets")
plt.ylabel("CVIR")
plt.title("Imbalance")
plt.legend()
plt.show()