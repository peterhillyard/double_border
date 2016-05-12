import numpy as np
import circ_buff_class as circBuffClass
import sys

#
# Author: Peter Hillyard
#
# Purpose: This class uses rss measurements from overlapping link lines to classify
# which section of the border the person crossed.  The class allows implements 
# the HMM forward classifier. AFter the object is created, observe() is used to
# add in a new rss vector and then get_state_est() gets which state the the 
# forward solution estimates. 
# 
# The states of any of the classifiers are one short segment being crossed or none 
# being crossed for state space S = {0,1,2,3,...,N-1} where N is the number of nodes.  
# We get a vector of RSS values and compute the emission probabilities using the 
# assumption of independence.

class myHmmBorderClass(object):
    
    # Input
    # A: Probability of transition in one step.  Matrix with A[i,j] as the
    #    one-step prob of transition from i to j.
    # pi: initial state probability.  Vector pi[i] is the probability of
    #    being in state i at time 0.
    # V: All possible observation values, eg., the alphabet
    # min_p: minimum probability allowed in the for the on and off link probabilities
    # p127: probability of measuring a 127 on a link
    # rss_obj: this object takes care of all the rss manipulation
    # ltb_len: length of the long term buffer for pmf updating
    # Delta: shift in mean between off and on pmf
    # eta: scalar multiplicative contanst on variance
    # omega: minimum allowable variance
    
    # Internal
    # num_links: number of links from the rss_obj
    # num_ch: number of channels from the rss_obj
    # on_links: holds the probabilities of observing RSS measurements for each link in the on state
    # off_links: holds the probabilities of observing RSS measurements for each link in the off state
    # codewords: The codewords for the off and on states
    # off_buff: long term buffer into which we add new RSS vector measurements
    # is_updated: 0 if in calibration period, 1 otherwise
    # off_count: number of samples that were in the off state
    # V_mat: The list of possible observations repeated by the number of links
    # num_states: The number of states in the Markov model
    # C: The circular buffer that stores the most recent emission probabilities for each state
    # alpha: A circular buffer containing the forward state probabilities for each time index

    def __init__(self, A, pi, V, min_p, p127, rss_obj, ltb_len, Delta, eta, omega):
        self.rss_obj = rss_obj
        self.num_links = rss_obj.get_L_sub()
        self.num_ch = rss_obj.get_C_sub()
        self.A = A
        self.pi = pi
        self.on_links = np.nan*np.ones((self.num_links,V.shape[0]))
        self.off_links = np.nan*np.ones((self.num_links,V.shape[0]))
        self.codewords = self.__get_codewords()
        self.min_p = min_p
        self.p127 = p127
        self.off_buff = circBuffClass.myCircBuff(ltb_len,self.num_links)
        self.is_updated = 0
        self.off_count = 0
        self.Delta = Delta
        self.eta = eta
        self.omega = omega
        
        self.V_mat = np.tile(V,(self.num_links,1))
        self.num_states = A.shape[0]
        self.C = circBuffClass.myCircBuff(2,A.shape[0])

        self.alpha = circBuffClass.myCircBuff(2,A.shape[0])

    #############################################
    #
    #
    #
    #
    # The following methods are to compute the most likely state in the hmm
    #
    #
    #
    #
    #############################################
        
    # This function takes the current RSS measurement from each link and 
    # converts it to the emission probabilities for each state
    def observe(self,cur_line):
        
        # Get the right RSS out
        self.rss_obj.observe(cur_line)
        cur_obs = self.rss_obj.get_rss()
        
        # if we are in calibration time, add the current observation to off 
        # buffer
        if np.logical_not(self.__is_ltb_full()):
            self.__add_obs_to_off_buff(cur_obs)
        
        # if we are done with calibration, and the pmfs have not been set, then
        # set them
        elif np.logical_not(self.is_updated):
            self.__set_static_gaus_pmfs()
            self.is_updated = 1
        
        # if we are done with calibration, and the pmfs are set, then go!
        if self.is_updated:
            
            # Get likelihoods of current vector observation
            self.__update_b_vec(cur_obs)
    
            # make a function call to update alpha
            self.__update_alpha()
            
            # update pmfs if necessary
            self.__update_pmfs(cur_obs)
            
    # get the most likely state based on the forward algorithm
    #
    def get_state_est(self):
        tmp = self.__get_state_probs()
        
        if np.isnan(np.sum(tmp)):
            return np.nan
        else:
            return np.argmax(tmp)
        
    # get the current rss time
    def get_time(self):
        return self.rss_obj.get_time()        
    
    ##############################################################
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # Observe helper functions
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ##############################################################
    # Add the current observation to the long term buffer
    def __add_obs_to_off_buff(self,cur_obs):
        self.off_buff.add_observation(cur_obs)
    
    # Check if long term buffer is full
    def __is_ltb_full(self):
        return self.off_buff.is_full()
            
    # Computes the observation given state probability
    def __update_b_vec(self,cur_obs):
        # convert measurement vector into emission probabilities
        # repeat the observation in columns
        cur_obs_mat = np.tile(cur_obs,(self.V_mat.shape[1],1)).T
        masked_mat = cur_obs_mat == self.V_mat

        # Extract the probability of the observation on each link for each state
        p_obs_given_off_link = np.sum(self.off_links*masked_mat,axis=1)
        p_obs_given_on_link  = np.sum(self.on_links*masked_mat,axis=1)

        # replicate the probability of each measurement on each link for each state
        p_obs_mat_off = np.tile(p_obs_given_off_link,(self.num_states,1)).T
        p_obs_mat_on  = np.tile(p_obs_given_on_link,(self.num_states,1)).T

        # Compute emission probabilities
        tmp1 = self.codewords*p_obs_mat_on
        tmp2 = np.logical_not(self.codewords)*p_obs_mat_off
        tmp3 = tmp1 + tmp2
        
        # divide tmp3 into groups of 4.  Multiply and normalize
        prev = np.ones(self.num_states)
        start_mark = 0
        end_mark = 4
        group = end_mark
        while start_mark < self.num_links:
            current = np.product(tmp3[start_mark:np.minimum(self.num_links,end_mark),:],axis=0)
            current = current/np.sum(current)
            prev = (prev*current)/np.sum(prev*current)
            end_mark += group
            start_mark += group

        # add emission probabilities to the circular buffer
        self.C.add_observation(prev)        
        
    # Compute the forward joint probability alpha.  Compute it for the most 
    #recent observation and add it to alpha's circular buffer
    def __update_alpha(self):
        # create the first alpha values when there is only one observation
        if self.C.get_num_in_buff() == 1:
            alphatmp = self.pi[:,0]*self.C.get_observation(1)
            self.alpha.add_observation(alphatmp/np.sum(alphatmp))
        
        # create the next alpha values when there is more than one observation
        else:
            alphatmp = np.dot(self.alpha.get_observation(1),self.A)*self.C.get_observation(1)
            self.alpha.add_observation(alphatmp/np.sum(alphatmp))
            
    # Update the pmfs if warrented 
    def __update_pmfs(self,cur_obs):
        
        tmp = self.__get_state_probs()
            
        # update the pmfs if we are in the off state for several samples
        if (np.isnan(tmp[0]) == 0) & (tmp[0] > 0.6):
            self.off_count += 1
            self.__add_obs_to_off_buff(cur_obs) # add observation to the buffer
            if self.off_count == 18:
                self.__set_static_gaus_pmfs()
                self.off_count = 0
        
    # Return the probabilities of being in each state
    def __get_state_probs(self):
        return 1*self.alpha.get_observation(1)
        
    
    ##################################################
    # 
    #
    #
    #
    # The following methods are used to set the pmfs for the links
    #
    #
    #
    #
    ##################################################
    
    # This method defines the on and off pmfs to be static gaussians where the 
    # on pmfs have a lower mean and larger variance
    def __set_static_gaus_pmfs(self):
        if np.logical_not(self.off_buff.is_full()):
            print "The long term buffer is not yet full.  This may give undesirable results"
        
        # median RSS of off-state buffer
        cal_med = self.off_buff.get_no_nan_median()
        
        if (np.sum(cal_med == 127) > 0) | (np.sum(np.isnan(cal_med)) > 0):
            sys.stderr.write('At least one link has a median of 127 or is nan\n\n')
            quit()
             
        if (np.sum(np.isnan(self.off_buff.get_nanvar())) > 0):
            sys.stderr.write('the long term buffer has a nan')
            quit()
        
        cal_med_mat = np.tile(cal_med,(self.V_mat.shape[1],1)).T
        
        # variance of RSS during calibration
        cal_var = np.maximum(self.off_buff.get_nanvar(),self.omega) #3.0 
        cal_var_mat = np.tile(cal_var,(self.V_mat.shape[1],1)).T
        
        # Compute the off_link emission probabilities for each link
        x = np.exp(- (self.V_mat - cal_med_mat)**2/(2*cal_var_mat/1.0)) # 1.0
        self.off_links = self.__normalize_pmf(x)
        
        # Compute the on_link emission probabilities for each link
        x = np.exp(- (self.V_mat - (cal_med_mat-self.Delta))**2/(self.eta*2*cal_var_mat)) # 3
        self.on_links = self.__normalize_pmf(x) 
    
    # This method takes a matrix where the rows represent an unscaled pmf for a 
    # given link.  It ensures that each pmf retains its shape and scaling values
    # to normalize the pmf to sum to 1 
    def __normalize_pmf(self,x):
        min_p = self.min_p
        p127 = self.p127
                  
        # indexes of where missed packets occur
        zero_idx = False*np.ones(x.shape)
        zero_idx[:,-1] = True
        
        # indexes of where x is less than the minimum allowable probability
        one_idx = x < min_p
        one_idx[:,-1] = False
        
        # indexes where x is above the min prob and not a missed packet
        two_idx = (zero_idx == 0) & (one_idx == 0)
        
        # normalizing parameter
        gamma = (1.0-np.sum(p127*zero_idx + min_p*one_idx,axis=1))/np.sum(x*two_idx,axis=1)
        
        # normalize only the points that are above the min
        x = p127*zero_idx + min_p*one_idx + np.tile(gamma,(self.V_mat.shape[1],1)).T*x*two_idx          
            
        return x
    
    
    ########################################################
    #
    #
    #
    #
    #
    #
    # Methods used in init stage
    #
    #
    #
    #
    #
    #
    ##########################################################
        
#     # This function produces the codewords matrix for a given number of nodes and channels
#     def __get_codewords(self,numNodes,numCh):
#         codewords_tmp = np.zeros((numNodes*(numNodes-1)/2,numNodes))
#         xing_pts = np.arange(-0.5,numNodes-0.5,1)
#         
#         linkLocs = np.array([(left,right) for left in range(0,numNodes-1) for right in range(0,numNodes) if (left != right) & (left < right)])
#          
#         for ii in range(xing_pts.shape[0]):
#             xing_pts_mat = np.tile(xing_pts[ii],(numNodes*(numNodes-1)/2))
#             codewords_tmp[:,ii] = 1.*((linkLocs[:,0] < xing_pts_mat) & (xing_pts_mat < linkLocs[:,1]))
#          
#         return np.tile(codewords_tmp,(numCh,1))
    
    # This function produces the codewords matrix for a given number of nodes and channels
    def __get_codewords(self):
        numNodes = self.rss_obj.get_N_sub()
        numCh = self.rss_obj.get_C_sub()
        
        codewords_tmp = np.zeros((self.num_links,numNodes))
        xing_pts = np.arange(-0.5,numNodes-0.5,1)
        
        if (self.rss_obj.fb_choice == 'f') | (self.rss_obj.fb_choice == 'b'):
            linkLocs = np.array([(tx,rx) for cc in range(numCh) for tx in range(0,numNodes-1) for rx in range(0,numNodes) if (tx != rx) & (tx < rx)])
        elif (self.rss_obj.fb_choice == 'fb'):
            linkLocs = np.array([(tx,rx) for cc in range(numCh*2) for tx in range(0,numNodes-1) for rx in range(0,numNodes) if (tx != rx) & (tx < rx)])
        elif (self.rss_obj.fb_choice == 'a'):
            linkLocs = np.array([(np.minimum(tx,rx),np.maximum(tx,rx)) for cc in range(numCh) for tx in range(0,numNodes) for rx in range(0,numNodes) if (tx != rx)])
            
         
        for ii in range(xing_pts.shape[0]):
            xing_pts_mat = np.tile(xing_pts[ii],self.num_links)
            codewords_tmp[:,ii] = 1.*((linkLocs[:,0] < xing_pts_mat) & (xing_pts_mat < linkLocs[:,1]))
         
        return codewords_tmp












