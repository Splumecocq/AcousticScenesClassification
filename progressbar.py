# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:15:11 2019

@author: Simon.Plumecocq

Simple progress bar

"""

import sys 
import time 
import numpy as np

def format_time(seconds): 
   days = int(seconds / 3600/24) 
   seconds = seconds - days*3600*24 
   hours = int(seconds / 3600) 
   seconds = seconds - hours*3600 
   minutes = int(seconds / 60) 
   seconds = seconds - minutes*60 
   secondsf = int(seconds) 
   seconds = seconds - secondsf 
   millis = int(seconds*1000) 
   f = '' 
   i = 1 
   if days > 0: 
      f += str(days) + 'D' 
      i += 1 
   if hours > 0 and i <= 2: 
      f += str(hours) + 'h' 
      i += 1 
   if minutes > 0 and i <= 2: 
      f += str(minutes) + 'm' 
      i += 1 
   if secondsf > 0 and i <= 2: 
      f += str(secondsf) + 's' 
      i += 1 
   '''   
   if millis > 0 and i <= 2: 
      f += str(millis) + 'ms' 
      i += 1 
   if f == '': 
      f = '0ms' 
   '''
   return f 

BAR_LENGTH = 30.0    
def progress_bar(current,end):
    global begin_time
    
    if current == 0:
        begin_time = time.time()
                  
    sys.stdout.write('\r')
    cur_len = int(BAR_LENGTH*(current+1)/end) 
    rest_len = int(BAR_LENGTH - cur_len)
    sys.stdout.write(' [') 
    for i in range(cur_len): 
      sys.stdout.write('#') 
    for i in range(rest_len): 
       sys.stdout.write('.') 
    sys.stdout.write(']') 

    step = np.round(100*(current+1)/end, decimals=0)
    sys.stdout.write(str(step)+"%")
    
    current_time = time.time() 
    global_time = format_time(current_time - begin_time)
    sys.stdout.write(" "+global_time)
    
    if current == end -1:
        sys.stdout.write(" Finish \n")       
    sys.stdout.flush() 

'''
#Exemple
nb_boucle = 10
for index_b in range(nb_boucle):
    time.sleep(0.5)
    progress_bar(index_b,nb_boucle)
'''  