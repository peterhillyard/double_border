
#! /usr/bin/env python

# This script reads packet data from the listen node through the serial port
# and prints one RSS measurement on each line for all tx, rx, ch combination,
# where the rx != tx, both rx and tx are in the list of sensors, and ch is in 
# the list of channels.
#
#
# Version History:
#
# Version 1.1:  Initial Release, 2 Oct 2012
# Version 1.2:  Includes data capture of RIP belts for respiratory monitoring, 25 June 2015

import sys
import serial
import time
import rss as rss


# What channels are measured by your nodes, in order of their measurement?
# USER:  SET THIS TO THE CHANNELS IN YOUR CHANNEL GROUP
channelList   = [26, 11, 16, 21]

# USER: The following serial "file name" changes depending on your operating 
#       system, and what name is assigned to the serial port when your listen 
#       node is plugged in.
serial_filename = rss.serialFileName()
sys.stderr.write('Using USB port file: ' + serial_filename + '\n')
ser = serial.Serial(serial_filename,38400)

# How many nodes the sensors have as the max # nodes (what # they're programmed with)
# USER:  THIS SHOULD NOT BE CHANGED, IT IS 6 FOR ALL GROUPS IN OUR CLASS
maxNodes      = int(sys.argv[1])

# What node numbers are yours, that you want to see output to the file.
# USER:  SET THIS TO THE NODE IDS ASSIGNED TO YOU.  DO NOT INCLUDE THE LISTEN NODE NUMBER
nodeList      = range(1,maxNodes+1)  # 1, ..., 30

# Parameters that are due to our implementation of the listen node.
numNodes      = len(nodeList)
numChs        = len(channelList)
numLinks      = numNodes*(numNodes-1)*numChs
rssIndex      = 3
string_length = maxNodes + 7
suffix        = ['ef','be']  # "0xBEEF"

# Initialize data, output file
nodeSet       = set(nodeList)
channelSet    = set(channelList)
currentLine   = []  # Init serial data buffer "currentLine" as empty.
currentLinkRSS = [127] * numLinks



# Run forever, adding one integer at a time from the serial port, 
#   whenever an integer is available.
while(1):
    tempInt = ser.read().encode('hex')
    currentLine.append(tempInt)

    # Whenever the end-of-line sequence is read, operate on the "packet" of data.
    if currentLine[-2:] == suffix:
        if len(currentLine) != string_length:
            sys.stderr.write('packet corrupted - wrong string length\n')
            del currentLine[:]
            continue
        currentLineInt = [int(x, 16) for x in currentLine]
        rxId = currentLineInt[2]
        currentCh = currentLineInt[-4]
        #print [currentLineInt[0:3], [hex2signedint(he) for he in currentLine[3:9]], currentLineInt[9:]]
        #print rxId
        if (rxId not in nodeSet) or (currentCh not in channelSet):
            del currentLine[:]
            continue                    
        
        # Each line in the serial data has RSS values for multiple txids.
        # Output one line per txid, rxid, ch combo.
        for txId in nodeList:
            # If the rxId is after the txId, then no problem -- currentCh
            # is also the channel that node txId was transmitting on when
            # node rxId made the measurement, because nodes transmit on a
            # channel in increasing order.
            if rxId > txId: 
                ch = currentCh
            else: 
                ch = rss.prevChannel(channelList, currentCh)
            
            # If the link (tx, rx, ch) is one we are supposed to watch
            if txId != rxId:  
                i = rss.linkNumForTxRxChLists(txId, rxId, ch, nodeList, channelList)
                
                # If the RSS has already been recorded for this link on 
                # this "line", then output the line first, and then restart 
                # with a new line.
                if currentLinkRSS[i] < 127:
                    # Output currentLinkRSS vector
                    sys.stdout.write(' '.join(map(str,currentLinkRSS)) + ' ' + str(time.time()) + '\n')
                    sys.stdout.flush()
                    # Restart with a new line by resetting currentLinkRSS
                    currentLinkRSS = [127] * numLinks
                
                # Store the RSS 
                currentLinkRSS[i] = rss.hex2signedint(currentLine[rssIndex+txId-1])

        # Remove serial data from the buffer.
        currentLine = []
