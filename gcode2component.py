
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

#from nptdms import TdmsFile
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dateutil import parser


#%%Global Initialized Variables
FOLDERNAME='UM3_polygon-3'
originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto/2018_DAC/dataCSV';

#The total channels from where data is collected
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
              
# variables to store the times for the each of the component
timesForComponent_1=[]
timesForComponent_2=[]
timesForComponent_3=[]
timesForComponent_4=[]
#%%
class gcodeParser:
    def __init__(self, gcodeString):
        self.G=np.NAN
        self.M=np.NAN
        self.F=np.NAN
        self.X=np.NAN
        self.Y=np.NAN
        self.Z=np.NAN
        self.E=np.NAN
        self.ALL=0
        try:
            if np.isnan(gcodeString):
                return
        except:
            temp=0;
            data_1=gcodeString.split(";")
            data_2=data_1[0].split()
            for data in data_2:
                
                if 'G' in data:
                    self.G=float(data.split("G")[1])
                    temp+=1;
                elif 'M' in data:
                    self.M=float(data.split("M")[1])
                    temp+=1;
                elif 'F' in data:
                    self.F=float(data.split("F")[1])
                    temp+=1;
                elif 'X' in data:
                    self.X=float(data.split("X")[1])
                    temp+=1;
                elif 'Y' in data:
                    self.Y=float(data.split("Y")[1])    
                    temp+=1;
                elif 'Z' in data:
                    self.Z=float(data.split("Z")[1])    
                    temp+=1;
                elif 'E' in data:
                    self.E=float(data.split("E")[1])    
                    temp+=1;
                else:
                    pass
                self.ALL=temp
#%%                
class lineSegment:
      def __init__(self,timeStart,timeStop, X1,Y1,Z1,E1,
                   X2,Y2,Z2,E2,components):
        self.timeStart=timeStart
        self.timeStop=timeStop
        self.X1=X1
        self.Y1=Y1
        self.X2=X2
        self.Y2=Y2
        self.Z1=Z1
        self.Z2=Z2
        self.E1=E1 #Extrusion amount
        self.E2=E2 #Extrusion amount
#        self.layer=layer
        self.component_0=components[0] #Vector storing which components
        self.component_1=components[1]
        self.component_2=components[2]
        self.component_3=components[3]
        #component1 = "StepperMotor_X"
        #component2 = "StepperMotor_Y"
        #component3 = "StepperMotor_Z"
        #component4 = "StepperMotor_E"
        
 #%%       
def gcode2linesegments(tdmsFolderNameFull):
    timingData = pd.read_csv(tdmsFolderNameFull+'/Timing.csv')
    lineSegmentList=[]
        
    previousTime=0;
    previousGcodeX=0;
    previousGcodeY=0
    previousGcodeZ=0;
    previousGcodeE=0;
    
#    layer=0;
    components=[False,False,False,False]
    
    
    startPrint=False
    
    for index, gcodeString in enumerate(timingData.GCode):
        gcode=gcodeParser(gcodeString)
        
        if not (startPrint):
            
            if np.isnan(gcode.X):
                    gcode.X=previousGcodeX
            if np.isnan(gcode.Y):
                    gcode.Y=previousGcodeY
            if np.isnan(gcode.Z):
                    gcode.Z=previousGcodeZ
            if np.isnan(gcode.E):
                    gcode.E=previousGcodeE        
                    
                        
            previousGcodeX=gcode.X;
            previousGcodeY=gcode.Y;
            previousGcodeZ=gcode.Z;
            previousGcodeE=gcode.E;
            previousTime= timingData.PCTime[index]
            
            # Ignore the initial data E becomes negative before printing begins
            if  gcode.M==107:
                startPrint=True;
                continue
       
        else:
            if gcode.E>0 and gcode.M==107: #printing stops 
                return lineSegmentList
                
            if gcode.G==1 or gcode.G==0:

                if np.isnan(gcode.X):
                    gcode.X=previousGcodeX
                    components[0]=False
                else:
                    if previousGcodeX==gcode.X:
                        components[0]=False
                    else:
                        components[0]=True
                
                if np.isnan(gcode.Y):
                    gcode.Y=previousGcodeY
                    components[1]=False
                else:
                    if previousGcodeY==gcode.Y:
                        components[1]=False
                    else:
                        components[1]=True
                    
                if np.isnan(gcode.Z):
                    gcode.Z=previousGcodeZ
                    components[2]=False
                else:
                    
                    if previousGcodeZ==gcode.Z:
                        components[2]=False
                    else:
                        components[2]=True
                        
#                        layer+=1
                
                if np.isnan(gcode.E):
                    gcode.E=previousGcodeE 
                    components[3]=False
                else:
                    if previousGcodeE==gcode.E:
                        components[3]=False
                    else:
                        components[3]=True
                
                
                lineSegmentList.append(lineSegment(previousTime,
                timingData.PCTime[index],previousGcodeX,
                previousGcodeY, previousGcodeZ,previousGcodeE, gcode.X,gcode.Y,
                gcode.Z, gcode.E,components))
#                print(components)
                previousGcodeX=gcode.X;
                previousGcodeY=gcode.Y;
                previousGcodeZ=gcode.Z;
                previousGcodeE=gcode.E;
                previousTime= timingData.PCTime[index]
  
            else:
               previousTime= timingData.PCTime[index] 
   
    return lineSegmentList            
         
#%% Plotting function   
def plotgcode(lineSegmentList):

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')      
#    color='r' 
    for coordinate in lineSegmentList:
#        print(coordinate.component_1, coordinate.component_2,coordinate.component_3,coordinate.component_4)
        if  coordinate.component_3==True:      
              ax.plot([coordinate.X1,coordinate.X2],
                      [coordinate.Y1,coordinate.Y2],
                      [coordinate.Z1,coordinate.Z2])
    plt.show()
#    plt.show()  
    plt.ioff()

         
      
#%% save the timing data based on components
    
def savetimingforcomponents(lineSegmentList,destinationFolderName):
    print("Saving Timing Data for each of the components...")
    component_0_T=[]
    component_0_F=[]
    component_1_T=[]
    component_1_F=[]
    component_2_T=[]
    component_2_F=[]
    component_3_T=[]
    component_3_F=[]
    
    for data in lineSegmentList: 
        timeStart = parser.parse(data.timeStart)
        timeStop = parser.parse(data.timeStop)
        if data.component_0==True:
            component_0_T.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond])
        else:
            component_0_F.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond])
        
        if data.component_1==True:
            component_1_T.append([timeStart.hour,timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond]) 
        else:
            component_1_F.append([timeStart.hour,timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond]) 
    
        if data.component_2==True:
            component_2_T.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond]) 
        else:
            component_2_F.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond])
    
        if data.component_3==True:
            component_3_T.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond]) 
        else:
            component_3_F.append([timeStart.hour, timeStart.minute,
                            timeStart.second,timeStart.microsecond,
                            timeStop.hour,timeStop.minute,
                            timeStop.second,timeStop.microsecond])
  
    filename=(destinationFolderName+'/Component_X.csv')
    np.savetxt(filename, component_0_T, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_no_X.csv')
    np.savetxt(filename, component_0_F, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_Y.csv')
    np.savetxt(filename, component_1_T, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_no_Y.csv')
    np.savetxt(filename, component_1_F, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_Z.csv')
    np.savetxt(filename, component_2_T, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_no_Z.csv')
    np.savetxt(filename, component_2_F, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_E.csv')
    np.savetxt(filename, component_3_T, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  
    
    filename=(destinationFolderName+'/Component_no_E.csv')
    np.savetxt(filename, component_3_F, delimiter=',', 
                   header='start_H, start_M, start_S, start_uS, stop_H, stop_M, stop_S, stop_uM'
                   , comments='')  

    print("Finished Saving Timing Data...")
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
    
#%% main file   
def gcode2component(tdmsFolderName):
    print("Starting Module...")
    tdmsFiles=[]
    tdmsFolderNameFull=originTDMSFolder+'/'+tdmsFolderName+'/data';
    
    if not os.path.exists(tdmsFolderNameFull):
        exit()
    
    for file in os.listdir(tdmsFolderNameFull):
        if file.endswith(".tdms"):
              tdmsFiles+= [file]
              
    # Check if the destination foldername exists
    destinationFolderName=destinationCSVFolder+'/'+tdmsFolderName;
    
    if not os.path.exists(destinationFolderName):
        os.makedirs(destinationFolderName)    
             
    # get the linesegmentlist from the timing data
    print("Parsing Gcode...")
    lineSegmentList=gcode2linesegments(tdmsFolderNameFull)
    
    print("plotting Graph...")
    plotgcode(lineSegmentList)
    
    savetimingforcomponents(lineSegmentList,destinationFolderName)
#    print("Stopping Module...")

#%% 
#if __name__=='__main__':
#    
#    tdmsFolderName = parsingInit()
#    
#    if not tdmsFolderName.isspace():
#        gcode2component(tdmsFolderName)
#    else:
#        print("...Module Stopped!")
    
  
