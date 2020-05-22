import numpy as np
import matplotlib.pyplot as plt
'''
GPAS_no_uncertainty = np.load('GPAS_no_uncertainty.npy')
clf_no_gp_no_uncertainty = np.load('clf_no_gp_no_uncertainty.npy')
pid_no_uncertainty = np.load('pid_no_uncertainty.npy')
ii = np.linspace(0,30,1500)
plt.figure()
plt.plot(ii[:400],clf_no_gp_no_uncertainty[:400],linewidth=3.0,label='CBF-CLF-QP')
plt.plot(ii[:400],GPAS_no_uncertainty,linewidth=2.0,linestyle='--',label='GPAS')
plt.plot(ii[:400],pid_no_uncertainty,linewidth=2.0,linestyle=':',label='PID')
plt.xlim([0,8])
plt.ylim([-0.2,1.2])
plt.legend(fontsize=15)

plt.xlabel('t/s')
plt.ylabel('V(t)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


clf_008_1_08 = np.load('clf_gamma1_008_gamma_2_1_epsilon_8.npy')
clf_002_1_08 = np.load('clf_gamma1_002_gamma_2_1_epsilon_08.npy')
clf_05_1_08 = np.load('clf_gamma1_05_gamma_2_1_epsilon_08.npy')
clf_008_01_08 = np.load('clf_gamma1_008_gamma_2_01_epsilon_08.npy')
clf_008_1_05 = np.load('clf_gamma1_008_gamma_2_1_epsilon_05.npy')
clf_008_1_02 = np.load('clf_gamma1_008_gamma_2_1_epsilon_02.npy')
clf_008_1_1 = np.load('clf_008_1_1.npy')
clf_0_1_08 = np.load('clf_0_1_08.npy')


ii = np.linspace(0,30,1500)
plt.figure()
plt.plot(ii,clf_002_1_08,linewidth=2.0,linestyle='--',label='clf_002_1_08------')
plt.plot(ii,clf_05_1_08,linewidth=2.0,linestyle='--',label='clf_05_1_08')
plt.plot(ii,clf_0_1_08,linewidth=2.0,linestyle='--',label='clf_0_1_08')

plt.plot(ii,clf_008_01_08,linewidth=2.0,linestyle=':',label='clf_008_01_08')

plt.plot(ii,clf_008_1_05,linewidth=2.0,linestyle='-.',label='clf_008_1_05')
plt.plot(ii,clf_008_1_02,linewidth=2.0,linestyle='-.',label='clf_008_1_02')
plt.plot(ii,clf_008_1_1,linewidth=2.0,linestyle='-.',label='clf_008_1_1')
plt.plot(ii,clf_008_1_08,linewidth=2.0,label='clf_008_1_08')

plt.xlim([0,30])
plt.ylim([-0.7,2.7])
plt.plot([10,10],[-0.5,2.5],c='k',linestyle='--')
plt.plot([20,20],[-0.5,2.5],c='k',linestyle='--')
plt.xlabel('t/s')
plt.ylabel('V(t)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.text(5, -0.5, "Phase 1",horizontalalignment='center', fontsize=15)
plt.text(15, -0.5, "Phase 2",horizontalalignment='center', fontsize=15)
plt.text(25, -0.5, "Phase 3",horizontalalignment='center', fontsize=15)

plt.legend(fontsize=15)
plt.show()

plt.figure()
clf_sense1 = np.load('clf.npy')
pid_sense1 = np.load('pid.npy')
clf_no_gp_sense1 = np.load('clf_no_gp.npy')
clf_no_lmp_sense1 = np.load('clf_no_lmp.npy')

GPAS_sense1 = np.load('GPAS.npy')
plt.plot(ii,clf_sense1,linewidth=2.0,label='CBF-CLF-QP')
plt.plot(ii,clf_no_gp_sense1,linewidth=2.0,label='CBF-CLF-QP without GP')
plt.plot(ii,clf_no_lmp_sense1,linewidth=2.0,label='CBF-CLF-QP without LMP')
plt.plot(ii,pid_sense1,linewidth=2.0,label='PID')
plt.plot(ii,GPAS_sense1,linewidth=2.0,label='GPAS')


plt.xlim([0,30])
plt.ylim([-0.25,1.1])
plt.plot([10,10],[-0.2,1.],c='k',linestyle='--')
plt.plot([20,20],[-0.2,1.],c='k',linestyle='--')
plt.xlabel('t/s')
plt.ylabel('V(t)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.text(5, -0.2, "Phase 1",horizontalalignment='center', fontsize=15)
plt.text(15, -0.2, "Phase 2",horizontalalignment='center', fontsize=15)
plt.text(25, -0.2, "Phase 3",horizontalalignment='center', fontsize=15)

plt.legend(fontsize=15)
plt.show()
'''
plt.figure()
ii = np.linspace(0,10,500)
clf_2 = np.load('clf_noise_-2.npy')
clf_3 = np.load('clf_noise_-1.npy')
clf_4 = np.load('clf_noise_-4.npy')
clf1 = np.load('clf_noise_-5.npy')
plt.plot(ii,clf_3,linewidth=3.0,label='[-1,-1,-1]')
plt.plot(ii,clf_2,linewidth=2.0,linestyle='--',label='[-2,-2,-2]')
plt.plot(ii,clf_4,linewidth=2.0,linestyle='-.',label='[-4,-4,-4]')
plt.plot(ii,clf1,linewidth=2.0,linestyle=':',label='[-5,-5,-5]')


plt.xlim([0,10])
plt.ylim([-0.25,2])
plt.xlabel('t/s')
plt.ylabel('V(t)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.legend(fontsize=15)
plt.show()

