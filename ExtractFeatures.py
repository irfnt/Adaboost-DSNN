"""
Extracts the desired feature set from the PHCX HTRU2 dataset. 

Run this script in a directory which has set of PHCX candidate files of one class. 
It outputs `out.csv` file with IDs, and 6 features as defined in Morello et. al. 

One may have to add the target column(`y') manually after extraction. 

"""
from phcx import Candidate
import numpy as np
import os
import fnmatch as fn

def scorer(s,b):
    if s >= 0:
        return (1 - np.exp(-s/b))
    else:
        return (s/b)

# list of things I want
# o self.snr
# o self.dm
# o self.width
# o self.bary_period
# o self.nsubs
# o self.nbins
# o self.fc
# o self.bw
# o self.profile
# o self.nbins_profile
fo = open('out.csv','w')
fo.write('id,one,two,three,four,five,six\n')
for file in os.listdir('.'):
    if fn.fnmatch(file, '*.phcx'):
        cid = file[7:-5]
        cand = Candidate(file) # I get an object
        one = np.log(cand.snr) # my first feature
        three = np.log(cand.bary_period/cand.dm) # my third feature
        four = np.tanh(cand.dm - 2) # my fourth feature
        ## For six
        rsm = 0.0 # My running sum
        for idx in range(cand.nbins_profile):
            for jdx in range(cand.nsubs):
                rsm = rsm + (cand.profile[idx] - cand.subints[jdx,idx])**2
        six = np.sqrt(rsm/(cand.width * cand.nsubs))
        ## for two
        smear_time = 8.3e-6 * (cand.bw) * ((cand.fc/1e3)**(-3)) * (cand.dm)
        # I've assumed some BW is Mhz and Fc is Ghz and dm is cm-3 pc
        two = (cand.width - smear_time)/cand.bary_period
        # Duty cycle w_eq is width ..P is bary_period
        ## for five
        bbb = (2*8)/np.sqrt(cand.nsubs)
        # I'm using the snr value only
        five = scorer(cand.snr,bbb)
        ## now I have all the features
        fo.write(cid+',')
        fo.write(str(one)+',')
        fo.write(str(two)+',')
        fo.write(str(three)+',')
        fo.write(str(four)+',')
        fo.write(str(five)+',')
        fo.write(str(six))
        fo.write('\n')
        # done with writing
## Close csv and save
fo.close()
