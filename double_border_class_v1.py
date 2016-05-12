import numpy as np
import point_class_v1 as aPointObj
import segment_class_v1 as aSegObj

# This class is used to get a path estimate of a person walking through a 
# double border.

class doubleBorder(object):
    
    
    def __init__(self, rss_editor, node_locs, out_type):
        
        self.rss_editor = rss_editor
        self.node_locs = node_locs
        self.link_line_network = []
        self.std_form_coefs = None
        self.midpoints = []
        self.classes = []
        self.codewords = np.zeros((self.rss_editor.get_L_sub(),(self.rss_editor.get_N_sub()/2-1)**2))
        self.out_type = out_type
        
        self.__get_network()
        self.__get_midpoints()
        self.__get_codewords()
    
    # 
    def observe(self, xed_link_lines, xing_times):
        
        # repeat binary vector history, and get the closest classification
        xed_mat = np.tile(xed_link_lines,(self.codewords.shape[1],1)).T
        hamming_dists = np.sum(xed_mat == (self.codewords == 1),axis=0)
        max_idx = hamming_dists == np.max(hamming_dists)
        
        if max_idx.sum() > 1:
            print "Still need to figure this out. quitting...\n"
            quit()
        
#         print self.classes[max_idx,:]
        
        # Get the Path Estimate
        classified_codeword = self.codewords[:,max_idx].flatten().astype(bool)
        xing_time_idx = np.logical_not(np.isnan(xing_times))
        master_idx = classified_codeword & xing_time_idx # indexes of only those links that are in the classified codeword and have non-nan time xings
        
        # extract times and line coefs
        std_form_coefs_sub = self.std_form_coefs[master_idx,:]
        time_vec_sub = xing_times[master_idx]
        time_vec_sub_zeroed = time_vec_sub - np.min(time_vec_sub)
        
        # Get constraining coefs and times
#         xing_time_min_idx = (xing_times == np.min(time_vec_sub)) & classified_codeword
#         xing_time_max_idx = (xing_times == np.max(time_vec_sub)) & classified_codeword
#         start_time = np.min(xing_times[xing_time_min_idx] - np.min(xing_times[xing_time_min_idx]))
#         end_time = np.max(xing_times[xing_time_max_idx] - np.min(xing_times[xing_time_min_idx]))
         
#         tmp1 = self.std_form_coefs[xing_time_min_idx]
#         tmp2 = self.std_form_coefs[xing_time_max_idx]
#         std_form_constrain = np.zeros((tmp1.shape[0]+tmp2.shape[0],3))
#         std_form_constrain[0:tmp1.shape[0],:] = tmp1
#         std_form_constrain[(tmp1.shape[0]):,:] = tmp2
#         start_time_vec = start_time*np.ones((tmp1.shape[0],1))
#         end_time_vec = end_time*np.ones((tmp2.shape[0],1))
#         start_end_time_vec = np.vstack((start_time_vec,end_time_vec)).flatten()
         
#         C_mat = np.zeros((std_form_constrain.shape[0],4))
#         C_mat[:,0] = std_form_constrain[:,0]*start_end_time_vec
#         C_mat[:,1] = std_form_constrain[:,0]
#         C_mat[:,2] = std_form_constrain[:,1]*start_end_time_vec
#         C_mat[:,3] = std_form_constrain[:,1]
#          
#         d_mat = std_form_constrain[:,2]
        
        # Form alpha matrix
        alpha_mat = np.zeros((time_vec_sub_zeroed.size,4))
        alpha_mat[:,0] = std_form_coefs_sub[:,0]*time_vec_sub_zeroed
        alpha_mat[:,1] = std_form_coefs_sub[:,0]
        alpha_mat[:,2] = std_form_coefs_sub[:,1]*time_vec_sub_zeroed
        alpha_mat[:,3] = std_form_coefs_sub[:,1]
        
        # if it is not full rank, we can't estimate reliably
        if np.linalg.matrix_rank(np.matrix(alpha_mat)) != 4:
            print "Not full rank.  Matrix inversion will be badly scaled."
        # otherwise continue estimation
        else:
            # Form Beta matrix
            beta_mat = std_form_coefs_sub[:,2]
            
            # Estimate Theta.  Theta is [v_x, p_x, v_y, p_y]^T
            theta = np.dot(np.linalg.inv(np.dot(alpha_mat.T,alpha_mat)), np.dot(alpha_mat.T,beta_mat.T))
            
            m = theta[2]/theta[0]
            b = theta[3] - theta[2]/theta[0]*theta[1]
            left_point = self.node_locs[-1,0]*m+b
            right_point = self.node_locs[0,0]*m+b
            
            p1 = aPointObj.Point(self.node_locs[-1,0],left_point)
            p2 = aPointObj.Point(self.node_locs[0,0],right_point)
            s1 = aSegObj.Segment(p1,p2)
            return s1
            
#             print("y=" + "%.2f" % m + "x+" + "%.2f" % b)
#             print("(-11.," + "%.2f" % left_point + ") <-> (11.," + "%.2f" % right_point + ")")
            
#             upl = 2.*np.dot(alpha_mat.T,alpha_mat)
#             upr = C_mat.T
#             lwl = C_mat
#             lwr = np.zeros((C_mat.shape[0],C_mat.shape[0]))
#              
#             upper_mat = np.hstack((upl,upr))
#             lower_mat = np.hstack((lwl,lwr))
#             full_mat = np.vstack((upper_mat,lower_mat))
#              
#             single_vec = np.hstack((2.0*np.dot(alpha_mat.T,beta_mat.T),d_mat))
#              
#             theta_all = np.dot(np.linalg.inv(full_mat),single_vec)
            
            
            
        
        
        
        
        
        
        
        
        
        
    
    # This method takes the node locations in the network and constructs a list
    # of line segment objects
    def __get_network(self):
        
        # Add the segment according to the network type: f, b, fb, a
        for ii in range(self.rss_editor.get_N_sub()):
            for jj in range(self.rss_editor.get_N_sub()):
                if (ii != jj):
                    if ii > jj:
                        if (self.rss_editor.fb_choice == 'a'):
                            point1 = aPointObj.Point(self.node_locs[ii,0],self.node_locs[ii,1])
                            point2 = aPointObj.Point(self.node_locs[jj,0],self.node_locs[jj,1])
                            
                            segment1 = aSegObj.Segment(point1,point2)
                            
                            self.link_line_network.append(segment1)
                        else:
                            continue
                    else:
                        point1 = aPointObj.Point(self.node_locs[ii,0],self.node_locs[ii,1])
                        point2 = aPointObj.Point(self.node_locs[jj,0],self.node_locs[jj,1])
                        
                        segment1 = aSegObj.Segment(point1,point2)
                        
                        self.link_line_network.append(segment1)
        
        if self.out_type == 'a':
            # copy for the fb type        
            if (self.rss_editor.fb_choice == 'fb'):
                self.link_line_network = self.link_line_network + self.link_line_network
            
            # copy the segments for the number of channels used if we consider all link lines            
            tmp = self.link_line_network[:]
            for cc in range(self.rss_editor.get_C_sub()-1):
                self.link_line_network += tmp
                
        # Get A, B, and C for standard form of a line
        self.std_form_coefs = np.zeros((len(self.link_line_network),3))
        
        for ii in range(len(self.link_line_network)):
            self.std_form_coefs[ii,:] = self.link_line_network[ii].get_std_coefs()
            
            
            
#         for ii in range(len(self.link_line_network)):
#             print '(' + str(self.link_line_network[ii].P1.x) + ',' + str(self.link_line_network[ii].P1.y) + ') -> (' + str(self.link_line_network[ii].P2.x) + ',' + str(self.link_line_network[ii].P2.y) + ')'
        
    # Get the midpoints between the nodes (exclude points between the first and last and the middle two)
    def __get_midpoints(self):
        
        tmp1 = []
        tmp2 = []
        
        for ii in range(self.rss_editor.get_N_sub()):
            if (ii+1 < self.rss_editor.get_N_sub()/2.0):
                point1 = aPointObj.Point(self.node_locs[ii,0],self.node_locs[ii,1])
                point2 = aPointObj.Point(self.node_locs[ii+1,0],self.node_locs[ii+1,1])
                segment1 = aSegObj.Segment(point1,point2)
                
                tmp1.append(segment1.get_midpoint())
            elif (ii+1) == self.rss_editor.get_N_sub()/2.0:
                continue
            elif (ii+1 == self.rss_editor.get_N_sub()):
                continue
            else:
                point1 = aPointObj.Point(self.node_locs[ii,0],self.node_locs[ii,1])
                point2 = aPointObj.Point(self.node_locs[ii+1,0],self.node_locs[ii+1,1])
                segment1 = aSegObj.Segment(point1,point2)
                
                tmp2.append(segment1.get_midpoint())
                
        self.midpoints = [tmp1, tmp2]
        
#         for row in [0,1]:
#             for col in [0,1,2]:
#                 print '(' + str(self.midpoints[row][col].x) + ',' + str(self.midpoints[row][col].y) + ')'
#             print '\n'
        
    # This method computes the codewords for all possible paths that go through 
    # opposite midpoints.  We check if the possible path intersects with the
    # link line segments in the network.
    def __get_codewords(self):
        
        nrows = len(self.link_line_network)
        ncols = len(self.midpoints[0])**2
        
        self.codewords = np.zeros((nrows,ncols))
        
        col_count = 0
        for ii in range(len(self.midpoints[0])):
            for jj in range(len(self.midpoints[0])):
                self.classes.append([ii,jj])
                for kk in range(len(self.link_line_network)):
                    A = self.midpoints[0][ii]
                    B = self.midpoints[1][jj]
                    C = self.link_line_network[kk].P1
                    D = self.link_line_network[kk].P2
                    
                    self.codewords[kk,col_count] = self.__intersect(A, B, C, D)
                col_count += 1
        
        self.classes = np.array(self.classes) + 1
        
    def __ccw(self,A,B,C):
        return (C.y-A.y) * (B.x-A.x) >= (B.y-A.y) * (C.x-A.x)

    # Return true if line segments AB and CD intersect
    def __intersect(self,A,B,C,D):
        return self.__ccw(A,C,D) != self.__ccw(B,C,D) and self.__ccw(A,B,C) != self.__ccw(A,B,D)
    

        