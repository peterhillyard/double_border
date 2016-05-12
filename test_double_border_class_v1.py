import numpy as np
import double_border_class_v1 as dbc
import rss_editor_class as anRSSeditor
import hmm_border_class_v1 as aHMMborder
import hmm_llcd_class_v1 as aHMMllcd
import matplotlib.pyplot as plt

####################
# Network parameters
numNodes = 8
numCh = 4

####################
# RSS data files and node locations
rss_data_f_name = 'data/rss_data/meb/rss_data_2015_10_19_all_2.txt'
node_locs = np.loadtxt('data/deployment_data/meb/node_locs.txt')

####################
# RSS editors
out_type = 'u'
fb_choice = 'fb'
ch_list = [0,1,2,3] # \in 0,1,2,...,C-1

# First border
node_list = [1,2,3,4] # \in 1,2,...,N
my_rss_editor_1 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

# Second border
node_list = [5,6,7,8] # \in 1,2,...,N
my_rss_editor_2 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

# Double Border
node_list = np.arange(numNodes)+1 # \in 1,2,...,N
my_rss_editor_3 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

# HMM LLCD
node_list = np.arange(numNodes)+1 # \in 1,2,...,N
my_rss_editor_4 = anRSSeditor.RssEditor(numNodes, numCh, ch_list, node_list, fb_choice)

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

#################
# HMM LLCD parameters
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
 
Delta = 7.0 # shift
eta = 4.0 # scale
omega = 0.75 # minimum variance
 
my_hmm_llcd = aHMMllcd.myHmmBorderClass(A,pi,V,min_p,p127,my_rss_editor_4,ltb_len,out_type,Delta,eta,omega)

#################
# Double border
my_dbc = dbc.doubleBorder(my_rss_editor_3, node_locs, out_type)

######################
# Loop through data
start_time = 0
end_time = 0
max_time = 15.

SF = 0
EF = 0

H1f = 0
H2f = 0

H1s = 0
H2s = 0

states_all_1 = []
states_all_2 = []
times_all = []
xing_times_est = []
out_seg_list = []

if out_type == 'u':
    bin_vec_history = np.zeros(my_hmm_llcd.num_links_unique).astype(bool)
    time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links_unique)]
elif out_type == 'a':
    bin_vec_history = np.zeros(my_hmm_llcd.num_links).astype(bool)
    time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links)]


with open(rss_data_f_name, 'r') as f:
    for line in f:
        
        my_hmmc_1.observe(line)
        my_hmmc_2.observe(line)
        my_hmm_llcd.observe(line)       
        
        H1 = my_hmmc_1.get_state_est() != 0
        H2 = my_hmmc_2.get_state_est() != 0    
        
        if not SF and not H1f and not H2f:
            if my_hmmc_1.get_time() > end_time + 2.0:
                if not H1 and H2:
                    SF = 1
                    H2f = 1
                    H2s = my_hmmc_2.get_state_est()
                    start_time = my_hmmc_1.get_time()
                if H1 and not H2:
                    SF = 1
                    H1f = 1
                    H1s = my_hmmc_1.get_state_est()
                    start_time = my_hmmc_1.get_time()
        
        # Check if time has elapsed
        elif SF and ((H1f and not H2 and not H2f) or (H2f and not H1 and not H1f)):
            if my_hmmc_1.get_time() - start_time > max_time:
                SF = 0
                H1f = 0
                H2f = 0
                
                H1s = 0
                H2s = 0
                
                if out_type == 'u':
                    bin_vec_history = np.zeros(my_hmm_llcd.num_links_unique).astype(bool)
                    time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links_unique)]
                elif out_type == 'a':
                    bin_vec_history = np.zeros(my_hmm_llcd.num_links).astype(bool)
                    time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links)]
                
        
        # The opposite border has been crossed
        elif (SF and not H1 and H1f and H2 and not H2f) or (SF and H1 and not H1f and not H2 and H2f):
            if H1s == 0:
                H1s = my_hmmc_1.get_state_est()
            elif H2s == 0:
                H2s = my_hmmc_2.get_state_est()
            
            # classification from two borders    
#             print H1s, H2s
            
            # OR the current binary vector            
            cur_time = my_hmmc_1.get_time()
            cur_bin_vec = my_hmm_llcd.get_state_est()
            bin_vec_history = bin_vec_history | cur_bin_vec
            
            # Get the median crossing time
            median_time_vec = np.zeros(cur_bin_vec.size)
            for ii in range(cur_bin_vec.size):
                if cur_bin_vec[ii] == 1:
                    time_vec_history[ii].append(cur_time)
                if len(time_vec_history[ii]) > 0:
                    median_time_vec[ii] = np.median(np.array(time_vec_history[ii]))
                else:
                    median_time_vec[ii] = np.nan

            end_time = cur_time
            
            # Get the classification and the 
            seg1 = my_dbc.observe(bin_vec_history, median_time_vec)
            out_seg_list.append(seg1)
            
            # Reset history
            if out_type == 'u':
                bin_vec_history = np.zeros(my_hmm_llcd.num_links_unique).astype(bool)
                time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links_unique)]
            elif out_type == 'a':
                bin_vec_history = np.zeros(my_hmm_llcd.num_links).astype(bool)
                time_vec_history = [[] for ll in xrange(my_hmm_llcd.num_links)] 
            
            xing_times_est.append(start_time)
            xing_times_est.append(end_time)
            
            SF = 0
            H1f = 0
            H2f = 0
            
            H1s = 0
            H2s = 0
            
        if SF:
            cur_time = my_hmmc_1.get_time()
            cur_bin_vec = my_hmm_llcd.get_state_est()
            bin_vec_history = bin_vec_history | cur_bin_vec
            
            for ii in range(cur_bin_vec.size):
                if cur_bin_vec[ii] == 1:
                    time_vec_history[ii].append(cur_time)
            
        states_all_1.append(my_hmmc_1.get_state_est())
        states_all_2.append(my_hmmc_2.get_state_est())
        times_all.append(my_hmmc_1.get_time())
            
# tmp = np.loadtxt('data/xing_data/meb/crossing_times_mod_2015_10_19.txt')
# print tmp[:,0]
# print xing_times_est

xing_times = np.loadtxt('data/xing_data/meb/crossing_times_mod_2015_10_19.txt') 
xing_locs = np.reshape(xing_times[:,1],(18,2))
mid_points = np.array([[11.,11.,11.,-11.,-11.,-11.],[25.,0.,-25.,-25.,0.,25.]]).T
midpoint_key = np.array([1.,2.,3.,4.,5.,6.])

for ii in range(xing_locs.shape[0]):
    
    # Truth
    xing_pt_1 = xing_locs[ii,0]
    xing_pt_2 = xing_locs[ii,1]
    mp1 = mid_points[midpoint_key == xing_pt_1,:].flatten()
    mp2 = mid_points[midpoint_key == xing_pt_2,:].flatten()
    
    plt.plot([out_seg_list[ii].P1.x,out_seg_list[ii].P2.x],[out_seg_list[ii].P1.y,out_seg_list[ii].P2.y],'b-.o',label='Estimate')
    
    plt.plot(node_locs[np.array([0,numNodes/2-1]),0],node_locs[np.array([0,numNodes/2-1]),1],'k-',lw=2)
    plt.plot(node_locs[np.array([numNodes-1,numNodes/2]),0],node_locs[np.array([numNodes-1,numNodes/2]),1],'k-',lw=2)
    plt.plot(node_locs[:,0],node_locs[:,1],'ro')
    plt.plot([mp1[0],mp2[0]],[mp1[1],mp2[1]],'g--x',label='Truth')
    plt.xlabel('Border Depth (ft)')
    plt.ylabel('Border Length (ft)')
    
    plt.axis([np.min(node_locs[:,0])-4,np.max(node_locs[:,0])+4,np.min(node_locs[:,1])-4,np.max(node_locs[:,1])+4])
    plt.grid()
    plt.legend()
    
    plt.show()
    
    plt.close()

quit()
        

times_all = np.array(times_all)
start_time = times_all[0]
times_all = times_all-start_time
xing_times_est = np.array(xing_times_est)
xing_times_est -= start_time
states_all_2 = np.array(states_all_2)
states_all_2[states_all_2 > 0] += 3
 
xing_times = np.loadtxt('data/xing_data/meb/crossing_times_mod_2015_10_19.txt') 
 
plt.plot(times_all,states_all_1,'r')
plt.plot(times_all,states_all_2,'k')
plt.plot(xing_times_est,1.5*np.ones(xing_times_est.size),'gx')
plt.plot(xing_times[:,0]-start_time,xing_times[:,1],'bx')
plt.ylim([0,numStates+3+2])
plt.grid()
plt.show()









