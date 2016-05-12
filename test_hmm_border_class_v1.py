import hmm_border_class_v1 as aHMMborder
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

node_list = [1,2,3,4] # \in 1,2,...,N
my_rss_editor_1 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

node_list = [5,6,7,8] # \in 1,2,...,N
my_rss_editor_2 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

#################
# HMMC parameters
numStates = my_rss_editor_1.get_N_sub()
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
 
Delta = 7.0 # shift
eta = 4.0 # scale
omega = 0.75 # minimum variance
 
my_hmmc_1 = aHMMborder.myHmmBorderClass(A,pi,V,min_p,p127,my_rss_editor_1,ltb_len,Delta,eta,omega)
my_hmmc_2 = aHMMborder.myHmmBorderClass(A,pi,V,min_p,p127,my_rss_editor_2,ltb_len,Delta,eta,omega)

######################
# Loop through data
states_all_1 = []
states_all_2 = []
times_all = []

with open(rss_data_f_name, 'r') as f:
    for line in f:
        
        my_hmmc_1.observe(line)
        my_hmmc_2.observe(line)
        
        states_all_1.append(my_hmmc_1.get_state_est())
        states_all_2.append(my_hmmc_2.get_state_est())
        times_all.append(my_hmmc_1.get_time())

times_all = np.array(times_all)
start_time = times_all[0]
times_all = times_all-start_time

xing_times = np.loadtxt('data/xing_data/meb/crossing_times_mod_2015_10_19.txt')
xing_times[xing_times[:,1] > 3,1] -= 3 

plt.plot(times_all,states_all_1,'r')
plt.plot(times_all,states_all_2,'k')
plt.plot(xing_times[:,0]-start_time,xing_times[:,1],'bx')
plt.ylim([0,numStates+3])
plt.grid()
plt.show()
        
        