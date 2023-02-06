# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:15:06 2020

@author: jacqu

Download box files from DUDE docking dir. 
"""

import urllib3
import os 

#targets = os.listdir('C:/Users/jacqu/Documents/mol2_resource/dude/all')
targets = ['hdac8']

for t in targets : 
    
    url = f'http://dude.docking.org/targets/{t}/docking/grids/box'
    
    # get the url 
    
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    print(r.data)
 
    with open('../data/conf/conf_{}_2.txt'.format(t), 'wb') as f:
        f.write(r.data)