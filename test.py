#%%
print("hello,world")

#%%
from matplotlib import pyplot as plt
import numpy as np

x = np.array([])
r = 0
for i in range(100000):
    a , b = np.random.rand() , np.random.rand()
    if a**2+b**2<=1 :
        r = r + 1
    x = np.insert(x,len(x),(r/(i+1))*4)

ax = plt.subplot()
ax.plot(x)
ax.set_ylim(3.13,3.15)
plt.show()

#%%
!dir