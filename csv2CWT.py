# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""
import numpy as np
import os
import argparse
import pandas as pd
import pywt
import time

#%%
FOLDERNAME='UM3_polygon-3'
CSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataCSV';
destinationFolderName = 'Z:/Google Drive/DT_Data/DAQ_Auto/2018_DAC/dataFeatures'
#destinationFolderName = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataFeatures'

component_T_names = ['Component_X.csv', 
                     'Component_Y.csv',
                     'Component_Z.csv',
                     'Component_E.csv']

channelNames=['Mic_1',
              'Mic_2',
              'Mic_3',
              'Mic_4',
               ];

#%%              
def parsingInit():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--foldername", type=str, nargs='?', 
                        default=" ",
                        help="Name of DAQ Folder")
    parser.add_argument("-c","--channelname", type=str, nargs='?', 
                        default=" ",
                        help="Name of of the channel")
    
    args = parser.parse_args()
    
    if not args.foldername.isspace():
        print ('Folder : ', args.foldername)
    else:
        print ('No Folder Name provided!')
        
    return  args.foldername,  args.channelname

 #%%       
def getcwt(fullfilepath):
   
   data=pd.read_csv(fullfilepath)
   
   widths = np.arange(1, 101)
   sig=data.values[:,0]
   
   cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh', 1/20000)
   
   cwtmags=abs(cwtmatr)
   
   return cwtmags

#%%
def splitFeature2Components(cwtmags, 
                            component_T_Start,
                            component_T_Stop,
                            currentSampleTime,
                            channelname,
                            csvFolderName):
 
    timeSteps=[]
    
    for i in range (0,np.shape(cwtmags)[1]):

        timeSteps.append(currentSampleTime+i*5e-05)
        
    component_T_indexes=[[] for j in range(4)]
    
    for index,times in enumerate(timeSteps):

        for i in range(4):
            for tStart,tStop in zip(component_T_Start[i],component_T_Stop[i]):
               
                if times<tStart:
                    break
                
                if times>=tStart and times<=tStop:
                    component_T_indexes[i].append(index)
                    break
                

    if not os.path.exists(destinationFolderName):
        os.makedirs(destinationFolderName)       
    
    objectDir= destinationFolderName+'/'+ csvFolderName 
    
    if not os.path.exists(objectDir):
        os.makedirs(objectDir)  
    
    channelDir= objectDir+'/'+channelname.split('.')[0] 
    
    if not os.path.exists(channelDir):
        os.makedirs(channelDir)
    
    
    for i in range(4):
        
#        print("Extracting Indexes for: ", component_T_names[i])
        component_T=[cwtmags[:,k] for k in component_T_indexes[i]] 

        
        filenameT= channelDir+'/'+ component_T_names[i]  

        
        component_T=np.array(component_T)

        with open(filenameT, mode="a") as file:
            df= pd.DataFrame(component_T)
            df.to_csv(file, sep=',', header=False, index=False)
            
#        print("Saved: ", component_T_names[i] )

    return timeSteps[-1]

#%%  
def csv2CWT(csvFolderName,gchannelname):
    
    csvFolderNameFull=CSVFolder+'/'+csvFolderName;
    
#    print (csvFolderNameFull)
    if not os.path.exists(csvFolderNameFull):
        print (csvFolderName, '-->Folder does not exist!')
        return
    
    startHour=pd.read_csv(csvFolderNameFull+'/Component_X.csv',header= 0, na_values='.').values[0,0]
    
    component_T_Start=[[] for i in range(4)]
    component_T_Stop=[[] for i in range(4)]
       
    for i in range(4):
        
        component_Temp = pd.read_csv(csvFolderNameFull+'/'+component_T_names[i])
#        print(len(component_Temp.values))
        for timess in component_Temp.values:
            startT=float(timess[0])*60*60+float(timess[1])*60+float(timess[2])+float(timess[3])*1e-6
            stopT=float(timess[4])*60*60+float(timess[5])*60+float(timess[6])+float(timess[7])*1e-6
            
            component_T_Start[i].append(startT)
            component_T_Stop[i].append(stopT)

 
    timeprintFlag=False
    datadirectories=[name for name in os.listdir(csvFolderNameFull) if "data" in str(name)]
    totaldirfiles=len(datadirectories)
    currentSampleTime=0
    
#    print(datadirectories,totaldirfiles)
    
    for datadirectory in datadirectories:
        print("")
        print(datadirectory)
        directory=csvFolderNameFull+'/'+datadirectory
        channels=[name for name in os.listdir(directory) if ".csv" in str(name)  if "timing" not in str(name)]
        
        start_time = time.time()
        timingMetaData= pd.read_csv(directory+'/timingMetaData.csv', header=None,  na_values='.')
        tM=timingMetaData.values[0,:]
#        print(tM)
        hour2s=float(startHour)*60*60
        min2s=float(tM[1])*60
        ss=float(tM[2])
        us2s=float(tM[3])*1e-6
        currentSampleTime=hour2s+min2s+ss+us2s
#        currentSampleTime=float(startHour)*60*60+float(tM[1])*60+float(tM[2])+float(tM[3])*1e-6
#        print(currentSampleTime)
        
        lastTime=0
        for channel in channels:
            
            if channel.split('.')[0] == gchannelname:
                print("Channel: ", channel)
                filepath=directory+'/'+channel
    
                cwtmags = getcwt(filepath)
                
                lastTime=splitFeature2Components(cwtmags, 
                                                  component_T_Start,
                                                  component_T_Stop,
                                                  currentSampleTime,
                                                  channel,
                                                  csvFolderName)
    
                if not timeprintFlag:
                  
                   print("Estimated Total Time for",  csvFolderName,
                         ":", totaldirfiles*(time.time() - start_time)/(60*60),
                         "Hours! \n")
                   timeprintFlag=True

        startHour=int(lastTime/3600)
        
    return      
       
#%%
if __name__=='__main__':
    
    foldername, gchannelname = parsingInit()
    
    if not foldername.isspace():
        csv2CWT(foldername, gchannelname)
    else:
        print("Default folder :",FOLDERNAME," Selected!")
        csv2CWT(FOLDERNAME)
        print("Module Stopping...")
 