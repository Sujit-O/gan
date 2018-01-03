import argparse
from tdms2csv import tdms2csv
from gcode2component import gcode2component

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
if __name__=='__main__':
    
    tdmsFolderName = parsingInit()
    
    if not tdmsFolderName.isspace():
        tdms2csv(tdmsFolderName)
        gcode2component(tdmsFolderName)
    else:
        print("Module Stopping...")
    