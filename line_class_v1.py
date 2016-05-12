class myLine(object):
    
    # Constructor
    #
    # a line is defined as ax+by=c
    def __init__(self, a=0., b=0., c=0.):
        self.a = a
        self.b = b
        self.c = c
        
        self.m = -a/float(b)
        self.yint = c/float(b)
        
    