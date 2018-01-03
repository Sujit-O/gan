import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os


dir1='out'
dir2='layer_3cn_900_itr_1000_tgs_2000_disitr_3_hiddim_128_bsize_64'
output='out/figure_3D_layer3'



def eval():
    if not os.path.exists(output):
                os.mkdir(output)
    hval=[0.2,0.4,0.6,0.8,1.0]
    for iteration in range(7):
        for c in range(3):
            print("C: ",c)
            for h in range(5):          
                print("H: ",h)
                df=pd.DataFrame()
                for i in range (100):
                    
                    filen='Distribution_Feature_'+str(i)+'_Condition_'+str(c)+'_H_'+str(hval[h])+'.csv'
                    
                    filename=dir1+'/'+dir2+'/'+filen
                    if not os.path.exists(filename):
                        print(filen, "does not exist!")
                        break
                    dfr=pd.read_csv(filename)
                    if i==0:
                        df=pd.DataFrame(dfr.values[iteration][1:-1])
                    else:
                        frame=[df, pd.DataFrame(dfr.values[iteration][1:-1])]
                        df=pd.concat(frame, axis=1)  
                
    #            print(np.shape(df))
                xydict=np.array(df.values)
#                print(np.shape(xydict))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X = np.arange(0, len(xydict))
                Y = np.arange(0, len(xydict[0]))
                X, Y = np.meshgrid(X, Y)
                Z = np.array(xydict).T
                
                ax.plot_surface(X[:,:-2], Y[:,:-2], Z[:,:-2], rstride=50, cstride=2, cmap=plt.cm.viridis)     
                # ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=plt.cm.hot)
                ax.set_zlim(0,3)
#                plt.show()
                
                output1=output+'/C_'+str(c)+'_H_'+str(h)
                if not os.path.exists(output1):
                        os.mkdir(output1)
                fig.savefig(output1+'/Iteration'+str(iteration)+'.png')   # save the figure to file
                plt.close(fig)
    return  


#%%
if __name__=="__main__":
    eval()