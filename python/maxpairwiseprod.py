#!/usr/bin/env python
# coding: utf-8

# In[5]:



def max2mult(a):
    
    if a[0]>a[1]:
        fir_max = a[0]
        snd_max = a[1]
    else:
        fir_max = a[1]
        snd_max = a[0]
    
    
    for i in range(len(a)): 
        if fir_max < a[i]:
            snd_max = fir_max
            fir_max = a[i]
        elif snd_max < a[i]:
            Snd_max = a[i]
   
    return fir_max * snd_max


# In[6]:


a = [2,9342,432,3484,4288,90000,3843412]


# In[7]:

print (max2mult(a))


# In[ ]:




