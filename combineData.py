# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:07:48 2017

@author: AICPS
"""
import numpy as np
import pandas as pd
import argparse
import os
import sys
#Global variables: path for the all the features, and path to store combined 

path_All_Features='D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataFeatures'
path_save_batch_Features='D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataCombined'
#path_All_Features='D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataFeatures'
#path_save_batch_Features='Z:/Google Drive/2018_DAC/data2'

#These are the only components we will use for this paper...sadly no time!
componentNames=['Component_X.csv',
                'Component_Y.csv',
                'Component_Z.csv',
                'Component_E.csv']

ydim=len(componentNames)
componentlabels=[np.zeros(shape=[1, ydim]),
                 np.zeros(shape=[1, ydim]),
                 np.zeros(shape=[1, ydim]),
                 np.zeros(shape=[1, ydim])]

for i,component in enumerate(componentlabels):
    componentlabels[i][0][i]=1


#%% create parser for the foldername, channel name and size
def parsingInit():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--foldername", type=str, nargs='?', 
                        default="UM3_polygon-4_2",
                        help="Name of DAQ Folder")
    parser.add_argument("-c","--channelname", type=str, nargs='?', 
                        default="Mic_4",
                        help="Name of of the channel")
    parser.add_argument("-s","--breakinto", type=int, nargs='?', 
                        default=2000,
                        help="break total data into how many chunks")
    
    args = parser.parse_args()
    
    if not args.foldername.isspace():
        print ('Folder provided : ', args.foldername)
        
    else:
        print ('Using Default Folder: ', args.foldername)
        
    if not args.channelname.isspace():
        print ('Channel provided: ', args.channelname)
        
    else:
        print ('Using Default Channel: ', args.channelname)
        
    print ('Data Chunks : ', args.breakinto)
 
    return  args.foldername,  args.channelname, args.breakinto

#%% the main function to combine the data 
def combineData(foldername, channelname, datachunks):
    folderpath=path_All_Features+'/'+foldername
    
    savingfilenames=['chunk'+str(i) for i in range(datachunks)]
    #check if the foldername exists!
    if not os.path.exists(folderpath):
        print(foldername,"does not exist!")
        print("Module stopped...")
        return
    
    #check if the channelfile exists!
    channelpath=folderpath+'/'+channelname
    #check if the foldername exists!
    if not os.path.exists(channelpath):
        print(channelname,"does not exist!")
        print("Module stopped...")
        return
    
    #check if all the component files exists!
    for filename in componentNames:
        componentfilepath=channelpath+'/'+filename
        if not os.path.exists(componentfilepath):
            print(filename,"does not exist!")
            print("Module stopped...")
            return
    #if all are present phew! lets start combining them
    
    
    if not os.path.exists(path_save_batch_Features):
        os.mkdir(path_save_batch_Features)
        
    
    destfolderpath=path_save_batch_Features+'/'+foldername
    if not os.path.exists(destfolderpath):
        os.mkdir(destfolderpath)
    
    destchannelpath=destfolderpath+'/' +channelname
    if not os.path.exists(destchannelpath):
        os.mkdir(destchannelpath)
    
    chunksize=[]
    for index, filename in enumerate(componentNames):
        
        componentfilepath=channelpath+'/'+filename
#        print("component filename:",filename)
        filesize = os.path.getsize(componentfilepath)
        df = pd.read_csv(componentfilepath, header=0, nrows=0,usecols=[0])
        datatypyesize=sys.getsizeof(df)
        row_count=float(filesize/(100*datatypyesize))
#        print("row_count:", row_count)
        chunksizeT=int(int(row_count)/datachunks)
        chunksize.append(chunksizeT)

    for i in range(datachunks):
        filename2save=destchannelpath+'/' +savingfilenames[i]+'.csv'
        print("chunk: ", savingfilenames[i])
        for j in range(3): #ignore the E for now
            componentfilepath=channelpath+'/'+componentNames[j]
#            print("component filename:",componentNames[j])
#            print(componentNames[j], i*chunksize[j])
#            continue
            df = pd.read_csv(componentfilepath, 
                                header=0,
                                skiprows=i*chunksize[j], 
                                nrows=chunksize[j])
                      
            
            y_df=[componentlabels[j][0] for i in range(df.shape[0])]
            y_df=pd.DataFrame(y_df)
            df=[df, y_df]
            df=pd.concat(df,axis=1)
            
            
            with open(filename2save, 'a') as f:
                df.to_csv(f, header=False)
   
    return
#%%
if __name__=="__main__":
    foldername, channelname, datachunks=parsingInit()
    combineData(foldername, channelname, datachunks)