import matplotlib.pyplot as plt
import numpy as np


# step=[18	18	18	18	18	18]
time=[1,1,1,14,88,27205]
node=[75,83,100,602,1485,25356]
my_x_ticks = np.arange(0, 0.6, 0.1)

# plot
plt.figure(2)
line1, =plt.plot(my_x_ticks,time,label='time',linestyle='-')
line2, =plt.plot(my_x_ticks,node,label='states',linestyle='--')
# line3, =plt.plot(val_loss,label='val_loss',linestyle='-.')
plt.legend(handles=[line1,line2], loc=2)
plt.xlabel('alpha')
plt.xticks(my_x_ticks)
plt.show()