import point_class_v1 as aPointObj
import numpy as np

class Segment(object):
    
    # constructor
    def __init__(self, P1 = None, P2=None):
        self.P1 = P1
        self.P2 = P2
        
    def get_midpoint(self):
        return aPointObj.Point((self.P1.x + self.P2.x)/2.0,(self.P1.y + self.P2.y)/2.0)
    
    def get_slope(self):
        
        if self.P1.x == self.P2.x:
            return np.nan
        else:
            return (self.P1.y-self.P2.y)/float(self.P1.x-self.P2.x)
        
    def get_y_int(self):
        m = self.get_slope()
        
        if np.isnan(m):
            return np.NaN
        else:
            return -m*self.P1.x+self.P1.y
        
    # This method returns the A, B, and C of a line in standard form
    # Ax + By = C where A>=0
    def get_std_coefs(self):
        
        # The Line is Vertical
        if self.P1.x == self.P2.x:
            A = self.P2.y-self.P1.y
            B = 0.
            C = (self.P2.y-self.P1.y)*self.P1.x
            if A>0:
                return np.array([A,B,C])
            else:
                return np.array([-A,B,-C])
        # The line is horizontal
        elif self.P1.y == self.P2.y:
            A=0.
            B=self.P1.x-self.P2.x
            C=(self.P1.x-self.P2.x)*self.P1.y            
            return np.array([A,B,C])
        else:
            A = self.P2.y-self.P1.y
            B = self.P1.x-self.P2.x
            C = (self.P2.y-self.P1.y)*self.P1.x + (self.P1.x-self.P2.x)*self.P1.y
            if A>0:
                return np.array([A,B,C])
            else:
                return np.array([-A,-B,-C])
            
        
        
        