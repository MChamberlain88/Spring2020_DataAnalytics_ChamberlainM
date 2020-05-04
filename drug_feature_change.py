# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:48:02 2020

@author: chambm6
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Import data

#Code to make heatmap correlation plots of features
dat = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/drug_consumption.data.csv")
plt.figure(figsize= (12, 12))
#Indexing columns
datcor = dat.iloc[:,1:13]
cor = datcor.corr()
ax = sns.heatmap(cor, annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#The following transformations are outlined on the data.world website
#I temporarily changed the numerical values of these categorical variables to their respective
#meanings so exploratory data analysis would be easier

#Converting age decimals to age buckets
for i in range(len(dat['age'])):
    if round(dat['age'][i],5) == -0.95197:
        dat['age'][i] = '18-24'
    elif round(dat['age'][i],5) == -0.07854:
        dat['age'][i] = '25-34'
    elif round(dat['age'][i],5) == .49788:
        dat['age'][i] = '35-44'
    elif round(dat['age'][i],5) == 1.09449:
        dat['age'][i] = '45-54'
    elif round(dat['age'][i],5) == 1.82213:
        dat['age'][i] = '55-64'
    elif round(dat['age'][i],5) == 2.59171:
        dat['age'][i] = '65+'
    else:
        print(dat['age'][i])
#Converting gender decimals to 0-female and 1-male
for i in range(len(dat['gender'])):
    if round(float(dat['gender'][i]),5) == .48246:
        dat['gender'][i] = int('0')
    elif round(float(dat['gender'][i]),5) == -.48246:
        dat['gender'][i] = int('1')
    else:
        print(dat['gender'][i])
#Transforming education decimals to descriptive educaltion level
for i in range(len(dat['education'])):
    if round(float(dat['education'][i]),5) == -2.43591:
        dat['education'][i] = 'Left School Before 16'
    elif round(float(dat['education'][i]),5) == -1.73790:
        dat['education'][i] = 'Left School at 16'
    elif round(float(dat['education'][i]),5) == -1.43719:
        dat['education'][i] = 'Left School at 17'
    elif round(float(dat['education'][i]),5) == -1.22751:
        dat['education'][i] = 'Left School at 18'
    elif round(float(dat['education'][i]),5) == -0.61113:
        dat['education'][i] = 'Some College, No Degree'
    elif round(float(dat['education'][i]),5) == -0.05921:
        dat['education'][i] = 'Professional Diploma'
    elif round(float(dat['education'][i]),5) == .45468:
        dat['education'][i] = 'University Degree'
    elif round(float(dat['education'][i]),5) == 1.16365:
        dat['education'][i] = 'Masters Degree'
    elif round(float(dat['education'][i]),5) == 1.98437:
        dat['education'][i] = 'Doctorate Degree'
    else:
        print(dat['education'][i])
##Changing country values to respective Cuntry name
for i in range(len(dat['country'])):
    if round(float(dat['country'][i]),5) == -.09765:
        dat['country'][i] = 'Austrailia'
    elif round(float(dat['country'][i]),5) == .24923:
        dat['country'][i] = 'Canada'
    elif round(float(dat['country'][i]),5) == -.46841:
        dat['country'][i] = 'New Zealand'
    elif round(float(dat['country'][i]),5) == -.28519:
        dat['country'][i] = 'Other'
    elif round(float(dat['country'][i]),5) == .21128:
        dat['country'][i] = 'Republic of Ireland'
    elif round(float(dat['country'][i]),5) == .96082:
        dat['country'][i] = 'UK'
    elif round(float(dat['country'][i]),5) == -.57009:
        dat['country'][i] = 'USA'
    else:
        print(dat['country'][i])
#Ethnicity encoding
for i in range(len(dat['ethnicity'])):
    if round(float(dat['ethnicity'][i]),5) == -.50212:
        dat['ethnicity'][i] = 'Aisian'
    elif round(float(dat['ethnicity'][i]),5) == -1.10702:
        dat['ethnicity'][i] = 'Black'
    elif round(float(dat['ethnicity'][i]),5) == 1.90725:
        dat['ethnicity'][i] = 'Mixed-Black/Asian'
    elif round(float(dat['ethnicity'][i]),5) == .12600:
        dat['ethnicity'][i] = 'Mixed-White/Asian'
    elif round(float(dat['ethnicity'][i]),5) == -.22166:
        dat['ethnicity'][i] = 'Mixed-White/Black'
    elif round(float(dat['ethnicity'][i]),5) == .11440:
        dat['ethnicity'][i] = 'Other'
    elif round(float(dat['ethnicity'][i]),5) == -.31685:
        dat['ethnicity'][i] = 'White'
    else:
        print(dat['ethnicity'][i])
 #Personality score encoding       
for i in range(len(dat['nscore'])):
    if round(float(dat['nscore'][i]),5) == -3.46436:
        dat['nscore'][i] = 12
    elif round(float(dat['nscore'][i]),5) == -3.15735:
        dat['nscore'][i] = 13
    elif round(float(dat['nscore'][i]),5) == -2.75696:
        dat['nscore'][i] = 14
    elif round(float(dat['nscore'][i]),5) == -2.52197:
        dat['nscore'][i] = 15
    elif round(float(dat['nscore'][i]),5) == -2.42317:
        dat['nscore'][i] = 16
    elif round(float(dat['nscore'][i]),5) == -2.34360:
        dat['nscore'][i] = 17
    elif round(float(dat['nscore'][i]),5) == -2.21844:
        dat['nscore'][i] = 18
    elif round(float(dat['nscore'][i]),5) == -2.05048:
        dat['nscore'][i] = 19
    elif round(float(dat['nscore'][i]),5) == -1.86962:
        dat['nscore'][i] = 20
    elif round(float(dat['nscore'][i]),5) == -1.69163:
        dat['nscore'][i] = 21
    elif round(float(dat['nscore'][i]),5) == -1.55078:
        dat['nscore'][i] = 22
    elif round(float(dat['nscore'][i]),5) == -1.43907:
        dat['nscore'][i] = 23
    elif round(float(dat['nscore'][i]),5) == -1.32828:
        dat['nscore'][i] = 24
    elif round(float(dat['nscore'][i]),5) == -1.19430:
        dat['nscore'][i] = 25
    elif round(float(dat['nscore'][i]),5) == -1.05308:
        dat['nscore'][i] = 26
    elif round(float(dat['nscore'][i]),5) == -.92104:
        dat['nscore'][i] = 27
    elif round(float(dat['nscore'][i]),5) == -.79151:
        dat['nscore'][i] = 28
    elif round(float(dat['nscore'][i]),5) == -.67825:
        dat['nscore'][i] = 29
    elif round(float(dat['nscore'][i]),5) == -.58016:
        dat['nscore'][i] = 30
    elif round(float(dat['nscore'][i]),5) == -.46725:
        dat['nscore'][i] = 31
    elif round(float(dat['nscore'][i]),5) == -.34799:
        dat['nscore'][i] = 32
    elif round(float(dat['nscore'][i]),5) == -.24649:
        dat['nscore'][i] = 33
    elif round(float(dat['nscore'][i]),5) == -.14882:
        dat['nscore'][i] = 34
    elif round(float(dat['nscore'][i]),5) == -.05188:
        dat['nscore'][i] = 35
    elif round(float(dat['nscore'][i]),5) == .04257:
        dat['nscore'][i] = 36
    elif round(float(dat['nscore'][i]),5) == .13606:
        dat['nscore'][i] = 37
    elif round(float(dat['nscore'][i]),5) == .22393:
        dat['nscore'][i] = 38
    elif round(float(dat['nscore'][i]),5) == .31287:
        dat['nscore'][i] = 39
    elif round(float(dat['nscore'][i]),5) == .41667:
        dat['nscore'][i] = 40
    elif round(float(dat['nscore'][i]),5) == .52135:
        dat['nscore'][i] = 41
    elif round(float(dat['nscore'][i]),5) == .62967:
        dat['nscore'][i] = 42
    elif round(float(dat['nscore'][i]),5) == .73545:
        dat['nscore'][i] = 43
    elif round(float(dat['nscore'][i]),5) == .82562:
        dat['nscore'][i] = 44
    elif round(float(dat['nscore'][i]),5) == .91093:
        dat['nscore'][i] = 45
    elif round(float(dat['nscore'][i]),5) == 1.02119:
        dat['nscore'][i] = 46
    elif round(float(dat['nscore'][i]),5) == 1.13281:
        dat['nscore'][i] = 47
    elif round(float(dat['nscore'][i]),5) == 1.23461:
        dat['nscore'][i] = 48
    elif round(float(dat['nscore'][i]),5) == 1.37297:
        dat['nscore'][i] = 49
    elif round(float(dat['nscore'][i]),5) == 1.49158:
        dat['nscore'][i] = 50
    elif round(float(dat['nscore'][i]),5) == 1.60383:
        dat['nscore'][i] = 51
    elif round(float(dat['nscore'][i]),5) == 1.72012:
        dat['nscore'][i] = 52
    elif round(float(dat['nscore'][i]),5) == 1.83990:
        dat['nscore'][i] = 53
    elif round(float(dat['nscore'][i]),5) == 1.98437:
        dat['nscore'][i] = 54
    elif round(float(dat['nscore'][i]),5) == 2.12700:
        dat['nscore'][i] = 55
    elif round(float(dat['nscore'][i]),5) == 2.28554:
        dat['nscore'][i] = 56
    elif round(float(dat['nscore'][i]),5) == 2.46262:
        dat['nscore'][i] = 57
    elif round(float(dat['nscore'][i]),5) == 2.61139:
        dat['nscore'][i] = 58
    elif round(float(dat['nscore'][i]),5) == 2.82196:
        dat['nscore'][i] = 59
    elif round(float(dat['nscore'][i]),5) == 3.27393:
        dat['nscore'][i] = 60
    else:
        print(dat['nscore'][i])
   
for i in range(len(dat['escore'])):
    if round(float(dat['escore'][i]),5) == -3.27493:
        dat['escore'][i] = 16
    elif round(float(dat['escore'][i]),5) == -3.00537:
        dat['escore'][i] = 17
    elif round(float(dat['escore'][i]),5) == -3.00537:
        dat['escore'][i] = 18
    elif round(float(dat['escore'][i]),5) == -2.72827:
        dat['escore'][i] = 19
    elif round(float(dat['escore'][i]),5) == -2.53830:
        dat['escore'][i] = 20
    elif round(float(dat['escore'][i]),5) == -2.44904:
        dat['escore'][i] = 21
    elif round(float(dat['escore'][i]),5) == -2.32338:
        dat['escore'][i] = 22
    elif round(float(dat['escore'][i]),5) == -2.21069:
        dat['escore'][i] = 23
    elif round(float(dat['escore'][i]),5) == -2.11437:
        dat['escore'][i] = 24
    elif round(float(dat['escore'][i]),5) == -2.03972:
        dat['escore'][i] = 25
    elif round(float(dat['escore'][i]),5) == -1.92173:
        dat['escore'][i] = 26
    elif round(float(dat['escore'][i]),5) == -1.76250:
        dat['escore'][i] = 27
    elif round(float(dat['escore'][i]),5) == -1.63340:
        dat['escore'][i] = 28
    elif round(float(dat['escore'][i]),5) == -1.50796:
        dat['escore'][i] = 29
    elif round(float(dat['escore'][i]),5) == -1.37639:
        dat['escore'][i] = 30
    elif round(float(dat['escore'][i]),5) == -1.23177:
        dat['escore'][i] = 31
    elif round(float(dat['escore'][i]),5) == -1.09207:
        dat['escore'][i] = 32
    elif round(float(dat['escore'][i]),5) == -.94779:
        dat['escore'][i] = 33
    elif round(float(dat['escore'][i]),5) == -.80615:
        dat['escore'][i] = 34
    elif round(float(dat['escore'][i]),5) == -.69509:
        dat['escore'][i] = 35
    elif round(float(dat['escore'][i]),5) == -.57545:
        dat['escore'][i] = 36
    elif round(float(dat['escore'][i]),5) == -.43999:
        dat['escore'][i] = 37
    elif round(float(dat['escore'][i]),5) == -.30033:
        dat['escore'][i] = 38
    elif round(float(dat['escore'][i]),5) == -.15487:
        dat['escore'][i] = 39
    elif round(float(dat['escore'][i]),5) == .00332:
        dat['escore'][i] = 40
    elif round(float(dat['escore'][i]),5) == .16767:
        dat['escore'][i] = 41
    elif round(float(dat['escore'][i]),5) == .32197:
        dat['escore'][i] = 42
    elif round(float(dat['escore'][i]),5) == .47617:
        dat['escore'][i] = 43
    elif round(float(dat['escore'][i]),5) == .63779:
        dat['escore'][i] = 44
    elif round(float(dat['escore'][i]),5) == .80523:
        dat['escore'][i] = 45
    elif round(float(dat['escore'][i]),5) == .96248:
        dat['escore'][i] = 46
    elif round(float(dat['escore'][i]),5) == 1.11406:
        dat['escore'][i] = 47
    elif round(float(dat['escore'][i]),5) == 1.28610:
        dat['escore'][i] = 48
    elif round(float(dat['escore'][i]),5) == 1.45421:
        dat['escore'][i] = 49
    elif round(float(dat['escore'][i]),5) == 1.58487:
        dat['escore'][i] = 50
    elif round(float(dat['escore'][i]),5) == 1.74091:
        dat['escore'][i] = 51
    elif round(float(dat['escore'][i]),5) == 1.93886:
        dat['escore'][i] = 52
    elif round(float(dat['escore'][i]),5) == 2.12700:
        dat['escore'][i] = 53
    elif round(float(dat['escore'][i]),5) == 2.32338:
        dat['escore'][i] = 54
    elif round(float(dat['escore'][i]),5) == 2.57309:
        dat['escore'][i] = 55
    elif round(float(dat['escore'][i]),5) == 2.85950:
        dat['escore'][i] = 56
    elif round(float(dat['escore'][i]),5) == 2.85950:
        dat['escore'][i] = 57
    elif round(float(dat['escore'][i]),5) == 3.00537:
        dat['escore'][i] = 58
    elif round(float(dat['escore'][i]),5) == 3.27393:
        dat['escore'][i] = 59
    else:
        print(dat['escore'][i]) 
    
         
        
for i in range(len(dat['oscore'])):
    if round(float(dat['oscore'][i]),5) == -3.27393:
        dat['oscore'][i] = 24
    elif round(float(dat['oscore'][i]),5) == -2.85950:
        dat['oscore'][i] = 26
    elif round(float(dat['oscore'][i]),5) == -2.63199:
        dat['oscore'][i] = 28
    elif round(float(dat['oscore'][i]),5) == -2.39883:
        dat['oscore'][i] = 29
    elif round(float(dat['oscore'][i]),5) == -2.21069:
        dat['oscore'][i] = 30
    elif round(float(dat['oscore'][i]),5) == -2.09015:
        dat['oscore'][i] = 31
    elif round(float(dat['oscore'][i]),5) == -1.97495:
        dat['oscore'][i] = 32
    elif round(float(dat['oscore'][i]),5) == -1.82919:
        dat['oscore'][i] = 33
    elif round(float(dat['oscore'][i]),5) == -1.68062:
        dat['oscore'][i] = 34
    elif round(float(dat['oscore'][i]),5) == -1.55521:
        dat['oscore'][i] = 35
    elif round(float(dat['oscore'][i]),5) == -1.42424:
        dat['oscore'][i] = 36
    elif round(float(dat['oscore'][i]),5) == -1.27553:
        dat['oscore'][i] = 37
    elif round(float(dat['oscore'][i]),5) == -1.11902:
        dat['oscore'][i] = 38
    elif round(float(dat['oscore'][i]),5) == -0.97631:
        dat['oscore'][i] = 39
    elif round(float(dat['oscore'][i]),5) == -.84732:
        dat['oscore'][i] = 40
    elif round(float(dat['oscore'][i]),5) == -.71727:
        dat['oscore'][i] = 41
    elif round(float(dat['oscore'][i]),5) == -.58331:
        dat['oscore'][i] = 42
    elif round(float(dat['oscore'][i]),5) == -.45174:
        dat['oscore'][i] = 43
    elif round(float(dat['oscore'][i]),5) == -.31776:
        dat['oscore'][i] = 44
    elif round(float(dat['oscore'][i]),5) == -.17779:
        dat['oscore'][i] = 45
    elif round(float(dat['oscore'][i]),5) == -.01928:
        dat['oscore'][i] = 46
    elif round(float(dat['oscore'][i]),5) == .14143:
        dat['oscore'][i] = 47
    elif round(float(dat['oscore'][i]),5) == .29338:
        dat['oscore'][i] = 48
    elif round(float(dat['oscore'][i]),5) == .44585:
        dat['oscore'][i] = 49
    elif round(float(dat['oscore'][i]),5) == .58331:
        dat['oscore'][i] = 50
    elif round(float(dat['oscore'][i]),5) == .72330:
        dat['oscore'][i] = 51
    elif round(float(dat['oscore'][i]),5) == .88309:
        dat['oscore'][i] = 52
    elif round(float(dat['oscore'][i]),5) == 1.06238:
        dat['oscore'][i] = 53
    elif round(float(dat['oscore'][i]),5) == 1.24033:
        dat['oscore'][i] = 54
    elif round(float(dat['oscore'][i]),5) == 1.43533:
        dat['oscore'][i] = 55
    elif round(float(dat['oscore'][i]),5) == 1.65653:
        dat['oscore'][i] = 56
    elif round(float(dat['oscore'][i]),5) == 1.88511:
        dat['oscore'][i] = 57
    elif round(float(dat['oscore'][i]),5) == 2.15324:
        dat['oscore'][i] = 58
    elif round(float(dat['oscore'][i]),5) == 2.44904:
        dat['oscore'][i] = 59
    elif round(float(dat['oscore'][i]),5) == 2.90161:
        dat['oscore'][i] = 60
    elif round(float(dat['oscore'][i]),5) == 'NaN':
        dat['oscore'][i] = 'NaN'
    
    else:
        print(dat['oscore'][i]) 
    
         

for i in range(len(dat['ascore'])):
    if round(float(dat['ascore'][i]),5) == -3.46436:
        dat['ascore'][i] = 12
    elif round(float(dat['ascore'][i]),5) == -3.15735:
        dat['ascore'][i] = 16
    elif round(float(dat['ascore'][i]),5) == -3.00537:
        dat['ascore'][i] = 18
    elif round(float(dat['ascore'][i]),5) == -2.90161:
        dat['ascore'][i] = 23
    elif round(float(dat['ascore'][i]),5) == -2.78793:
        dat['ascore'][i] = 24
    elif round(float(dat['ascore'][i]),5) == -2.70172:
        dat['ascore'][i] = 25
    elif round(float(dat['ascore'][i]),5) == -2.53830:
        dat['ascore'][i] = 26
    elif round(float(dat['ascore'][i]),5) == -2.35413:
        dat['ascore'][i] = 27
    elif round(float(dat['ascore'][i]),5) == -2.21844:
        dat['ascore'][i] = 28
    elif round(float(dat['ascore'][i]),5) == -2.07848:
        dat['ascore'][i] = 29
    elif round(float(dat['ascore'][i]),5) == -1.92595:
        dat['ascore'][i] = 30
    elif round(float(dat['ascore'][i]),5) == -1.77200:
        dat['ascore'][i] = 31
    elif round(float(dat['ascore'][i]),5) == -1.62090:
        dat['ascore'][i] = 32
    elif round(float(dat['ascore'][i]),5) == -1.47955:
        dat['ascore'][i] = 33
    elif round(float(dat['ascore'][i]),5) == -1.34289:
        dat['ascore'][i] = 34
    elif round(float(dat['ascore'][i]),5) == -1.21213:
        dat['ascore'][i] = 35
    elif round(float(dat['ascore'][i]),5) == -1.07533:
        dat['ascore'][i] = 36
    elif round(float(dat['ascore'][i]),5) == -.91699:
        dat['ascore'][i] = 37
    elif round(float(dat['ascore'][i]),5) == -.76096:
        dat['ascore'][i] = 38
    elif round(float(dat['ascore'][i]),5) == -.60633:
        dat['ascore'][i] = 39
    elif round(float(dat['ascore'][i]),5) == -.45321:
        dat['ascore'][i] = 40
    elif round(float(dat['ascore'][i]),5) == -.30172:
        dat['ascore'][i] = 41
    elif round(float(dat['ascore'][i]),5) == -.15487:
        dat['ascore'][i] = 42
    elif round(float(dat['ascore'][i]),5) == -.01729:
        dat['ascore'][i] = 43
    elif round(float(dat['ascore'][i]),5) == .13136:
        dat['ascore'][i] = 44
    elif round(float(dat['ascore'][i]),5) == .28783:
        dat['ascore'][i] = 45
    elif round(float(dat['ascore'][i]),5) == .43852:
        dat['ascore'][i] = 46
    elif round(float(dat['ascore'][i]),5) == .59042:
        dat['ascore'][i] = 47
    elif round(float(dat['ascore'][i]),5) == .76096:
        dat['ascore'][i] = 48
    elif round(float(dat['ascore'][i]),5) == .94156:
        dat['ascore'][i] = 49
    elif round(float(dat['ascore'][i]),5) == 1.11406:
        dat['ascore'][i] = 50
    elif round(float(dat['ascore'][i]),5) == 1.2861:
        dat['ascore'][i] = 51
    elif round(float(dat['ascore'][i]),5) == 1.45039:
        dat['ascore'][i] = 52
    elif round(float(dat['ascore'][i]),5) == 1.61108:
        dat['ascore'][i] = 53
    elif round(float(dat['ascore'][i]),5) == 1.81866:
        dat['ascore'][i] = 54
    elif round(float(dat['ascore'][i]),5) == 2.03972:
        dat['ascore'][i] = 55
    elif round(float(dat['ascore'][i]),5) == 2.23427:
        dat['ascore'][i] = 56
    elif round(float(dat['ascore'][i]),5) == 2.46262:
        dat['ascore'][i] = 57
    elif round(float(dat['ascore'][i]),5) == 2.75696:
        dat['ascore'][i] = 58
    elif round(float(dat['ascore'][i]),5) == 3.15735:
        dat['ascore'][i] = 59
    elif round(float(dat['ascore'][i]),5) == 3.46436:
        dat['ascore'][i] = 60
    elif round(float(dat['ascore'][i]),5) == 'NaN':
        dat['ascore'][i] = 'NaN'
    else:
        print(dat['ascore'][i]) 
 

for i in range(len(dat['cscore'])):
    if round(float(dat['cscore'][i]),5) == -3.46436:
        dat['cscore'][i] = 17
    elif round(float(dat['cscore'][i]),5) == -3.15735:
        dat['cscore'][i] = 19
    elif round(float(dat['cscore'][i]),5) == -2.90161:
        dat['cscore'][i] = 20
    elif round(float(dat['cscore'][i]),5) == -2.72827:
        dat['cscore'][i] = 21
    elif round(float(dat['cscore'][i]),5) == -2.57309:
        dat['cscore'][i] = 22
    elif round(float(dat['cscore'][i]),5) == -2.42317:
        dat['cscore'][i] = 23
    elif round(float(dat['cscore'][i]),5) == -2.30408:
        dat['cscore'][i] = 24
    elif round(float(dat['cscore'][i]),5) == -2.18109:
        dat['cscore'][i] = 25
    elif round(float(dat['cscore'][i]),5) == -2.04506:
        dat['cscore'][i] = 26
    elif round(float(dat['cscore'][i]),5) == -1.92173:
        dat['cscore'][i] = 27
    elif round(float(dat['cscore'][i]),5) == -1.78169:
        dat['cscore'][i] = 28
    elif round(float(dat['cscore'][i]),5) == -1.64101:
        dat['cscore'][i] = 29
    elif round(float(dat['cscore'][i]),5) == -1.51840:
        dat['cscore'][i] = 30
    elif round(float(dat['cscore'][i]),5) == -1.38502:
        dat['cscore'][i] = 31
    elif round(float(dat['cscore'][i]),5) == -1.25773:
        dat['cscore'][i] = 32
    elif round(float(dat['cscore'][i]),5) == -1.13788:
        dat['cscore'][i] = 33
    elif round(float(dat['cscore'][i]),5) == -1.01450:
        dat['cscore'][i] = 34
    elif round(float(dat['cscore'][i]),5) == -.89891:
        dat['cscore'][i] = 35
    elif round(float(dat['cscore'][i]),5) == -.78155:
        dat['cscore'][i] = 36
    elif round(float(dat['cscore'][i]),5) == -.65253:
        dat['cscore'][i] = 37
    elif round(float(dat['cscore'][i]),5) == -.52745:
        dat['cscore'][i] = 38
    elif round(float(dat['cscore'][i]),5) == -.40581:
        dat['cscore'][i] = 39
    elif round(float(dat['cscore'][i]),5) == -.27607:
        dat['cscore'][i] = 40
    elif round(float(dat['cscore'][i]),5) == -.14277:
        dat['cscore'][i] = 41
    elif round(float(dat['cscore'][i]),5) == -.00665:
        dat['cscore'][i] = 42
    elif round(float(dat['cscore'][i]),5) == .12331:
        dat['cscore'][i] = 43
    elif round(float(dat['cscore'][i]),5) == .25953:
        dat['cscore'][i] = 44
    elif round(float(dat['cscore'][i]),5) == .41594:
        dat['cscore'][i] = 45
    elif round(float(dat['cscore'][i]),5) == .58489:
        dat['cscore'][i] = 46
    elif round(float(dat['cscore'][i]),4) == .7583:
        dat['cscore'][i] = 47
    elif round(float(dat['cscore'][i]),5) == .93949:
        dat['cscore'][i] = 48
    elif round(float(dat['cscore'][i]),5) == 1.13407:
        dat['cscore'][i] = 49
    elif round(float(dat['cscore'][i]),5) == 1.30612:
        dat['cscore'][i] = 50
    elif round(float(dat['cscore'][i]),5) == 1.46191:
        dat['cscore'][i] = 51
    elif round(float(dat['cscore'][i]),5) == 1.63088:
        dat['cscore'][i] = 52
    elif round(float(dat['cscore'][i]),5) == 1.81175:
        dat['cscore'][i] = 53
    elif round(float(dat['cscore'][i]),5) == 2.04506:
        dat['cscore'][i] = 54
    elif round(float(dat['cscore'][i]),5) == 2.33337:
        dat['cscore'][i] = 55
    elif round(float(dat['cscore'][i]),5) == 2.63199:
        dat['cscore'][i] = 56
    elif round(float(dat['cscore'][i]),5) == 3.00537:
        dat['cscore'][i] = 57
    elif round(float(dat['cscore'][i]),5) == 3.46436:
        dat['cscore'][i] = 59
    elif round(float(dat['cscore'][i]),5) == 'NaN':
        dat['cscore'][i] = 'NaN'
    else:
        print(dat['cscore'][i]) 
#Frequency of drug use encoding       
for i in dat:
    for j in range(len(dat[i])):
        if dat[i][j] == 'CL0':
            dat[i][j] = 'Never Used'
        elif dat[i][j] == 'CL1':
            dat[i][j] = 'Used over a Decade Ago'
        elif dat[i][j] == 'CL2':
            dat[i][j] = 'Used in Last Decade'
        elif dat[i][j] == 'CL3':
            dat[i][j] = 'Used in Last Year'
        elif dat[i][j] == 'CL4':
            dat[i][j] = 'Used in Last Month'
        elif dat[i][j] == 'CL5':
            dat[i][j] = 'Used in Last Week'
        elif dat[i][j] == 'CL6':
            dat[i][j] = 'Used in Last Day'
for i in range(len(dat['gender'])):
    if dat['gender'][i] == 1:
        dat['gender'][i] = 'Male'
    elif dat['gender'][i] == 0:
        dat['gender'][i] = 'Female'
       
            
dat.to_csv('C:/Users/chambm6/Desktop/Data_Analytics/eda.csv')
     
        
         
        
        
        
        
        
        
    

    

    