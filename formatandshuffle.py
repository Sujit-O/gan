import numpy as np
import csv
import argparse
import os
import pandas as pd

START_CN        = 150
DATA_FILE_NAME  = "D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataCombined" # data file path
Destination     = "D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/finaltestData"
FOLDER_NAMES    = ['UM3_polygon-4_2',
                   'UM3_polygon-4_1',
                   'UM3_polygon-4'] #add 3D objects for futher testing
CHANNEL         = "Mic_4" #only testing microphone channel, there are 25 in total!
chunkcount      =  300
zncount         =  0
def parsingInit():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-fn","--foldernumber", type=int, nargs='?', 
                        default=3,  #default = UM3_polygon-4_2 
                        help="Total folders to take into consideration")
    parser.add_argument("-cn","--chunknumber", type=int, nargs='?', 
                        default=200,
                        help="Total chunks to take from each folder")
    

    args = parser.parse_args()
    
    print ('Folder Number : ', args.foldernumber)
    for i in range( args.foldernumber):
        print("Folder ",i," :",FOLDER_NAMES[i])
    print ('Chunk Number : ', args.chunknumber)
    
 
    return  (args.foldernumber,  
             args.chunknumber)
    
    
def formatandshuffle(fn,cn):
    global chunkcount
      
    subsample=10
    filepath=DATA_FILE_NAME+'/'+ FOLDER_NAMES[fn]+'/'+CHANNEL+'/chunk'+str(cn)+'.csv'
    
    z_df = pd.DataFrame()
    for zi in range (20):
        filepath2=DATA_FILE_NAME+'/'+ FOLDER_NAMES[fn]+'/'+CHANNEL+'/chunk'+str(cn+zi)+'.csv'
       
        with open(filepath2) as f:
            df1 = pd.read_csv(f)
                        
        data=np.array(df1.values) 
       
    
#        print(np.shape(data))
        data=data[np.where(data[:,-2] == 1)] #changed here! to remove E!
#        print(np.shape(data))
        rown=0
        new_row=int(np.shape(data)[0]/subsample)
        datan=np.zeros((new_row, np.shape(data)[1]))
        for i in range(new_row):
             datan[i,:]=np.mean(data[rown:rown+subsample, :], axis=0)
             rown+=subsample 
#        print(np.shape(datan)) 
        
        
        if zi==0:
            z_df=pd.DataFrame(datan) 
        
        else:   
            df=pd.DataFrame(datan)
            frame=[z_df, df]
            z_df=pd.concat(frame)
           
    
    with open(filepath) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            x = list(csvReader)
            data = np.array(x).astype("float32")
            
            
    data=data[np.where(data[:,-1] == 0)] #changed here! to remove E!
    rown=0
    new_row=int(np.shape(data)[0]/subsample)
    datan=np.zeros((new_row, np.shape(data)[1]))
    for i in range(new_row):
         datan[i,:]=np.mean(data[rown:rown+subsample, :], axis=0)
         rown+=subsample
    
    frames= [z_df, pd.DataFrame(datan)]   
    z_df=pd.concat(frames)
    
    datan=np.array(z_df.values) 
       
    perm = np.arange(len(datan))
       
    np.random.shuffle(perm)
    
    # create a matrix with randomized rows
    rand_data = datan[perm]
    
    df=pd.DataFrame(rand_data)
    if not os.path.exists(Destination):
            os.makedirs(Destination) 
    
    filesavepath=Destination+'/chunk'+str(chunkcount)+'.csv'
    chunkcount+=1
    
    with open(filesavepath,'w') as f:
      df.to_csv(f, header=False, index=False, sep=',') 
      
    return  



if __name__=="__main__":
    global data 
    fn,cn=parsingInit()
    for foldernumber in range(fn):
        print("foldername:",FOLDER_NAMES[foldernumber] )
        for chunknumber in range(cn):
            print("chunk:",chunknumber+START_CN)
            formatandshuffle(foldernumber,chunknumber+START_CN)
          