# This script takes the files created by the timestamp app on an iphone and 
# creates a modified file that includes a "seconds since last epoch" and the
# segment crossed.

import numpy as np
from datetime import timedelta
import datetime
import time

######################################
# Get true times
######################################
ss = 'crossing'
file_name_in = 'data/xing_data/meb/original_timestamps/' + ss + '_times_2015_10_19.txt'
file_name_out = 'data/xing_data/meb/' + ss + '_times_mod_2015_10_19.txt'

# file_name_in = '/home/pete/Copy/crossing_detection/data/xing_data/park/' + ss + '_times_2015_07_23.txt'
# file_name_out = '/home/pete/Copy/crossing_detection/data/xing_data/park/' + ss + '_times_mod_2015_07_23.txt'

f_xing = open(file_name_in)

num_lines = sum(1 for line in f_xing)

f_xing = open(file_name_in)

line_num = 1

time_seg = np.zeros((num_lines-3,2))
        
# loop through all training times
for line in f_xing:
    if (line_num==1) | (line_num==3):
        line_num += 1
        continue
    
    if line_num == 2:
        cur_line = line.split(' ')
        
        if cur_line[1] == '+':
            delt_t = timedelta(minutes=int(cur_line[2]),seconds=int(cur_line[4]),microseconds=int(cur_line[6]))
        else:
            delt_t = -timedelta(minutes=int(cur_line[2]),seconds=int(cur_line[4]),microseconds=int(cur_line[6]))
#         print delt_t
        line_num += 1
        continue
    
    cur_seg = int(line.split(',')[-1][0])
    cur_dt = line.split(',')[0]
    dt = datetime.datetime.strptime(cur_dt, "%Y/%m/%d %H:%M:%S.%f") + delt_t
    cur_time = time.mktime(dt.timetuple())
    time_seg[line_num-4,:] = cur_time, cur_seg 
    line_num += 1

ind = np.lexsort((time_seg[:,1],time_seg[:,0]))

np.savetxt(file_name_out, time_seg[ind]) 