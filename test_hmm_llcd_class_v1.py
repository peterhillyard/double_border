import hmm_llcd_class_v1 as aHMMllcd
import rss_editor_class as anRSSeditor
import numpy as np
import matplotlib.pyplot as plt

####################
# Network parameters
numNodes = 8
numCh = 4

####################
# RSS data files
rss_data_f_name = 'data/rss_data/meb/rss_data_2015_10_19_all_2.txt'

####################
# RSS editors
fb_choice = 'fb'
ch_list = [0,1,2,3] # \in 0,1,2,...,C-1

node_list = np.arange(numNodes)+1 # \in 1,2,...,N
my_rss_editor_1 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

#################
# HMMC parameters
numStates = 2
num_links = my_rss_editor_1.get_L_sub()
numCh = my_rss_editor_1.get_C_sub()

A = np.zeros((numStates,numStates))
A[0,0] = 0.8
A[0,1:] = (1-A[0,0])/(numStates-1)
A[1:,0] = 0.6
for ii in range(1,int(numStates)):
    A[ii,ii] = 1.0-A[ii,0]
pi = np.zeros((numStates,1))
pi[0,0] = 0.9
pi[1:,0] = (1-pi[0,0])/(numStates-1)
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 51 # length of the long term buffer
out_type = 'a'
 
Delta = 7.0 # shift
eta = 4.0 # scale
omega = 0.75 # minimum variance

my_hmmc_1 = aHMMllcd.myHmmBorderClass(A,pi,V,min_p,p127,my_rss_editor_1,ltb_len,out_type,Delta,eta,omega)

######################
# Loop through data
states_all = []
times_all = []

with open(rss_data_f_name, 'r') as f:
    for line in f:
        
        my_hmmc_1.observe(line)
        states_all.append(my_hmmc_1.get_state_est().tolist())
        times_all.append(my_hmmc_1.get_time())
        
states_all = np.array(states_all)
times_all = np.array(times_all)
times_all = times_all - times_all[0]

# plt.plot(times_all,states_all[:,0],'r-')
# plt.plot(times_all,states_all[:,1],'k--')
# plt.plot(times_all,states_all[:,2],'g-.')

plt.plot(times_all,states_all[:,3],'r-')
plt.plot(times_all,states_all[:,4],'k--')
plt.plot(times_all,states_all[:,5],'g-.')
plt.plot(times_all,states_all[:,6],'m-')
plt.ylim([0,1.5])
plt.grid()
plt.show()
        