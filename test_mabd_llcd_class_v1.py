import numpy as np
import mabd_llcd_class_v1 as AmabdLLCD
import rss_editor_class as anRSSeditor
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
# MABD parameters
stb_len = 5  # short term buffer length
ltb_len = 20 # long term buffer length
stateb_len = 12 # state buffer length
tau = 0.025   # threshold
num_consec = 2
out_type = 'a'

my_mabd = AmabdLLCD.myMABDClass(my_rss_editor_1, stb_len, ltb_len, stateb_len, tau, num_consec, out_type)

######################
# Loop through data
states_all = []
times_all = []

with open(rss_data_f_name, 'r') as f:
    for line in f:
        
        my_mabd.observe(line)
        states_all.append(my_mabd.get_state_est().tolist())
        times_all.append(my_mabd.get_time())
        
states_all = np.array(states_all)
times_all = np.array(times_all)
times_all = times_all - times_all[0]

plt.plot(times_all,states_all[:,0],'r-')
plt.plot(times_all,states_all[:,1],'k--')
plt.plot(times_all,states_all[:,2],'g-.')

# plt.plot(times_all,states_all[:,3],'r-')
# plt.plot(times_all,states_all[:,4],'k--')
# plt.plot(times_all,states_all[:,5],'g-.')
# plt.plot(times_all,states_all[:,6],'m-')
plt.ylim([0,1.5])
plt.grid()
plt.show()