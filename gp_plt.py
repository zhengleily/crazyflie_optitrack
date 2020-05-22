import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
f = np.load('f.npy')
g = np.load('g.npy').reshape([-1,9,4])
std = np.load('std.npy')
obs = np.load('obs.npy')
action = np.load('action.npy')
mean = np.load('mean.npy')
plt.figure()
ii = np.linspace(0,20,1000)
dt = 0.02
index = 5
start = 10
true_err = (obs[start+1:,index]-(f[start:-1,index]+ np.squeeze(np.matmul(g[start:-1,:,:],action[start:-1,:].reshape([-1,4,1]))[:,index,:])- mean[start:-1,index]))/dt
prediction = mean[start:-1,index]/0.02
up = (mean+3*std)/dt
down = (mean-3*std)/dt

plt.fill_between(ii[start+1:],down[start:-1,index],up[start:-1,index],color='k',alpha=.2,label='Uncertainty')
plt.plot(ii[start+1:],true_err,c='blue',linewidth=2.0,label='Actual Model Error')
plt.plot(ii[start+1:],prediction,c='red',linewidth=2.0,label='GPR Prediction')
plt.xlabel('t/s')
plt.ylabel('Error/m')
plt.ylim([-4.5,0.])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15,loc=4)

plt.show()

plt.figure()
index = 5
true_err = (obs[51:,index]-(f[50:-1,index]+ np.squeeze(np.matmul(g[50:-1,:,:],action[50:-1,:].reshape([-1,4,1]))[:,index,:])- mean[50:-1,index]))
err_rate = np.abs((f[50:-1,index]+ np.squeeze(np.matmul(g[50:-1,:,:],action[50:-1,:].reshape([-1,4,1]))[:,index,:])- mean[50:-1,index]))
plt.plot(err_rate)
#plt.show()




plt.figure()
index = 5
true_err = (obs[51:,index]-(f[50:-1,index]+ np.squeeze(np.matmul(g[50:-1,:,:],action[50:-1,:].reshape([-1,4,1]))[:,index,:])- mean[50:-1,index]))
err_rate = np.abs(true_err/(f[50:-1,index]+ np.squeeze(np.matmul(g[50:-1,:,:],action[50:-1,:].reshape([-1,4,1]))[:,index,:])- obs[50:-1,index]))
def to_percent(temp, position):
    return '%.2f'%(100 * temp) + '%'
plt.plot(err_rate)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#plt.show()
