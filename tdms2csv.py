# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

from nptdms import TdmsFile
import numpy as np
import os
import argparse

FOLDERNAME='UM3_polygon-5'

originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataCSV';

channelNames=['Mic_1',
              'Mic_2',
              'Mic_3',
              'Mic_4',
              'Current',
              'Vib_x2',
              'Vib_y2',
              'Vib_z2',
              'Vib_x1',
              'Vib_y1',
              'Vib_z1',
              'Vib_x0',
              'Vib_y0',
              'Vib_z0',
              'Temperature',
              'Humidity',
              'Mag_x0',
              'Mag_y0',
              'Mag_z0',
              'Mag_x1',
              'Mag_y1',
              'Mag_z1',
              'Mag_x2',
              'Mag_y2',
              'Mag_z2' ];

#%%              
def parsingInit():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--foldername", type=str, nargs='?', 
                        default=" ",
                        help="Name of DAQ Folder")
    
    args = parser.parse_args()
    
    if not args.foldername.isspace():
        print ('Folder : ', args.foldername)
    else:
        print ('No Folder Name provided!')
        
    return  args.foldername
        

 
#%%  
def tdms2csv(tdmsFolderName):
    tdmsFiles=[]
    tdmsFolderNameFull=originTDMSFolder+'/'+tdmsFolderName+'/data';
    
    if not os.path.exists(tdmsFolderNameFull):
        print (tdmsFolderName, '-->Folder does not exist!')
        return
    
    for file in os.listdir(tdmsFolderNameFull):
        if file.endswith(".tdms"):
              tdmsFiles+= [file]
    #%% Check if the destination foldername exists
    destinationFolderName=destinationCSVFolder+'/'+tdmsFolderName;
    
    if not os.path.exists(destinationFolderName):
        os.makedirs(destinationFolderName)    
             
    #%%
    directoryIndex=1;
    for file in tdmsFiles: 
        print(file)
        tdms_file = TdmsFile(tdmsFolderNameFull+'/'+file)
        
        directory=destinationCSVFolder+'/'+tdmsFolderName+'/data_'+str(directoryIndex);
        directoryIndex+=1;
        
        if not os.path.exists(directory):
            os.makedirs(directory) 
        
        for channelName in channelNames:
            channel = tdms_file.object('data',channelName)
            data=channel.data
            s=channel.property('wf_start_time')
            samplingIncrement=channel.property('wf_increment')
            
            np.savetxt(directory+'/'+channelName+'.csv', data, delimiter=',')
            np.savetxt(directory+'/timingMetaData.csv', [[s.hour,s.minute,s.second,s.microsecond, samplingIncrement]], delimiter=',')
            
#%%
#if __name__=='__main__':
#    
#    foldername = parsingInit()
#    
#    if not foldername.isspace():
#        tdms2csv(foldername)
#    else:
#        print("...Module Stopped!")
#    
    