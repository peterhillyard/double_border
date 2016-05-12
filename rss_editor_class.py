import numpy as np

##############################################
# A class for manipulating lines of RSS from linkAllLinks.py
#
# When using listenAllLinks.py, forward and backward links are all saved
# in the following order: (1,2,ch1), (1,3,ch1), ... (1,N,ch1), (2,1,ch1), 
# (2,3,ch1), ... (N,N-1,ch1), (1,2,ch2), ... , (N,N-1,chC)
# There may be occasions when we want only the forward links or only the 
# backward links instead of all of the links.  The following functions take as 
# input a line of RSS values and returns either the forward or the backward 
# link-channels.  The user can also specify which nodes and channels to include.
class RssEditor:
    # Constructor:
    
    # N_tot: total number of nodes used during data collection
    # C_tot: total number of channels used during data collection
    # num_links_tot: number of links formed between all N_tot nodes and C_tot channels
    # fb_choice: The type of links desired.  A user can choose 'f' for forward, 
    #            'b' for backward, 'fb' for forward then backward (in that order),
    #            or 'a' for all.
    # ch_list: a list that contains which channels the user desires.
    # node_list: a list of the nodes the user wants to measure on.
    
    # N_sub: The number of nodes in the node_list
    # C_sub: The number of channels in the ch_list
    # num_links_sub: the number of links measured on based on the fb_choice,
    #                the node_list, and the ch_list
    # cur_line_all: the current string from the .txt file
    # cur_rss_all: the current rss from the .txt file
    # cur_rss_sub: the rss measurements based on the fb_choice, the node_list,
    #              and the ch_list
    # cur_time: the timestamp associated with the current line from .txt
    # fw_idx: a binary vector that stores the indexes of ALL forward links.  This 
    #         vector does not take node_list or ch_list into account
    # bw_idx: a binary vector that stores the indexes of ALL backward links.  This
    #         vector does not take node_list or ch_list into account
    # aw_idx: a binary vector that stores the indexes of ALL links.  THis vector
    #         does not take node_list or ch_list into account
    # master_fw_idx: a binary vector that stores the indexes of the forward links
    #                taking into account the node_list and ch_list
    # master_bw_idx: a binary vector that stores the indexes of the backward links
    #                taking into account the node_list and ch_list
    # master_aw_idx: a binary vector that stores the indexes of all links taking 
    #                into account the node_list and ch_list
    # master_bw_ints: a vector that stores the indexes of the backward links taking
    #                 into account the node_list and ch_list
    # ch_idx: a binary index that indicates the indexes corresponding to the 
    #         channels desired by the user
    # link_idx: a binary vector that indicates which links are valid according
    #           to the node_list
    
    # get_idx: Get the master indexes so that we can reformat the full rss vector
    #          to get the desired RSS subset 
    
    def __init__(self, numNodes, numCh, ch_list, node_list, fb_choice):
        self.N_tot = numNodes
        self.C_tot = numCh
        self.num_links_tot = numCh*numNodes*(numNodes-1)
        
        self.fb_choice = fb_choice
        self.ch_list = ch_list
        self.node_list = node_list
        
        self.N_sub = len(node_list)
        self.C_sub = len(ch_list)
        self.num_links_sub = 0
        
        self.cur_line_all = None
        self.cur_rss_all = None
        self.cur_rss_sub = None
        self.cur_time = None
        
        self.fw_idx, self.bw_idx, self.aw_idx = (None, None, None)
        self.master_fw_idx, self.master_bw_idx, self.master_aw_idx = (None, None, None)
        self.master_bw_ints = None
        self.ch_idx = None
        self.link_idx = None
        
        self.get_idx()
    
    ############
    # Methods - We assume that rss_line is a numpy array
    ############
    
    # This takes a current line (as a string) from the file and parses it into
    # rss and time
    def observe(self,line):
        self.cur_line_all = line
        lineList         = [float(i) for i in line.split()]
        self.cur_time    = lineList.pop(-1)  # remove last element
        self.cur_rss_all = np.array(lineList) # get all rss values
        self.set_rss_subset()       
    
    # Return to the user the rss values requested
    def get_rss(self):
        return self.cur_rss_sub
    
    # Return the number of nodes from the node list    
    def get_N_sub(self):
        return self.N_sub
    
    # Return the number of channels from the channel list
    def get_C_sub(self):
        return self.C_sub
    
    # Return the number of links the user wants rss measurements from
    def get_L_sub(self):
        return self.num_links_sub
    
    # Return the current time
    def get_time(self):
        return self.cur_time
    
    # Return a desired rss value
    def get_rss_val(self,val):
        return self.cur_rss_sub[val]
        
        
    ###############
    # Helper methods
    ###############
    # Set only the RSS values the user specifies   
    def set_rss_subset(self):
        if self.fb_choice == 'f':
            self.cur_rss_sub = self.cur_rss_all[self.master_fw_idx]
        elif self.fb_choice == 'b':
            self.cur_rss_sub = self.cur_rss_all[self.master_bw_ints]
        elif self.fb_choice == 'fb':
            self.cur_rss_sub = np.append(self.cur_rss_all[self.master_fw_idx], self.cur_rss_all[self.master_bw_ints])
        elif self.fb_choice == 'a':
            self.cur_rss_sub = self.cur_rss_all[self.master_aw_idx]
    
    # Get the indexes corresponding to user specifications
    def get_idx(self):
        linkLocs = 1.+np.array([(tx,rx) for cc in range(self.C_tot) for tx in range(0,self.N_tot) for rx in range(0,self.N_tot) if (tx != rx)])
        
        # Get indexes of forward and backward links
        self.fw_idx = linkLocs[:,0] < linkLocs[:,1]
        self.bw_idx = np.logical_not(self.fw_idx)
        self.aw_idx = np.ones(linkLocs.shape[0]) == 1
        
        # Get channel indexes
        self.ch_idx = np.zeros(self.num_links_tot)
        for cc in self.ch_list:
            self.ch_idx[self.num_links_tot/self.C_tot*cc + np.arange(self.num_links_tot/self.C_tot)] = 1
        self.ch_idx = (self.ch_idx == 1)
            
        # Get link indexes
        tx_idx = np.zeros(self.num_links_tot)
        rx_idx = np.zeros(self.num_links_tot)
        for nn in self.node_list:
            tx_idx += (nn == linkLocs[:,0])
            rx_idx += (nn == linkLocs[:,1])
        self.link_idx = (tx_idx == 1) & (rx_idx == 1)
        
        # create master index array
        self.master_fw_idx = self.fw_idx & self.ch_idx & self.link_idx
        self.master_bw_idx = self.bw_idx & self.ch_idx & self.link_idx
        self.master_aw_idx = self.aw_idx & self.ch_idx & self.link_idx
        
        # Get the index of the backward links
        tmp = np.zeros((linkLocs.shape[0],3))
        tmp[:,0:2] = linkLocs
        tmp[:,-1] = np.arange(linkLocs.shape[0])
        tmp2 = tmp[self.master_bw_idx,:]
        
        tmp4 = [[] for cc in range(self.C_sub)]
        
        for nn in self.node_list:
            tmp3 = tmp2[tmp2[:,1] == nn]
            val = tmp3.shape[0]/self.C_sub
            for cc in range(self.C_sub):
                tmp4[cc] += tmp3[val*cc+np.arange(val),2].tolist()
                
        self.master_bw_ints = np.array(tmp4).flatten().astype(int)
        
        # calculate the number of subset links
        if self.fb_choice == 'f':
            self.num_links_sub = self.master_fw_idx.sum()
        elif self.fb_choice == 'b':
            self.num_links_sub = self.master_bw_idx.sum()
        elif self.fb_choice == 'fb':
            self.num_links_sub = self.master_fw_idx.sum() + self.master_bw_idx.sum()
        elif self.fb_choice == 'a':
            self.num_links_sub = self.master_aw_idx.sum()
            
        
        
        
        
        
        
        
        
        