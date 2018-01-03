import pandas as pd
import os
import pickle
import numpy as np

class hyperparameters:
    def __init__(self,layer=2, hiddenode=128, Z_dim=100,
    X_dim=100, Y_dim=4, outdir='../out', inputdir='../data',
    minibatch=64, chunknumber=900,
    iteration_Total=1000, testsamplegen=1000,
    gen_itr=2, dis_itr=2, saveFigure=False,
    savedistro=False, evalSec=True,saveIterNumber=10, loadhparam=False):
        if loadhparam:
            self.outdir             =   outdir
            self.loadhparam()
        else:
            self.layer              =   layer
            self.hiddenode          =   hiddenode
            self.Z_dim              =   Z_dim
            self.X_dim              =   X_dim
            self.Y_dim              =   Y_dim
            self.outdir             =   outdir
            self.inputdir           =   inputdir
            self.minibatch          =   minibatch
            self.chunknumber        =   chunknumber
            self.iteration_Total    =   iteration_Total
            self.testsamplegen      =   testsamplegen
            self.gen_itr            =   gen_itr
            self.dis_itr            =   dis_itr
            self.saveFigure         =   saveFigure
            self.savedistro         =   savedistro
            self.evalSec            =   evalSec
            self.saveIterNumber     =   saveIterNumber


    def printhparam(self):
        print ('********************Hyper Parameters****************')
        print ('                  Layer Number : ', self.layer)
        print ('           Hidden Nodes Number : ', self.hiddenode)
        print ('                    Input size : ', self.X_dim)
        print ('       Condition Encoding size : ', self.Y_dim)
        print ('                   Z dimension : ', self.Z_dim)
        print ('Discriminator iteration number : ', self.dis_itr)
        print ('    Generator iteration number : ', self.gen_itr)
        print ('        Test Gen Sample Number : ', self.testsamplegen)
        print ('           Training Batch Size : ', self.minibatch)
        print ('              Iteration Number : ', self.iteration_Total)
        print ('        Input DataChunk Number : ', self.chunknumber)
        print ('***************************************************\n')

        print ('********************I/O Directories****************')
        print ('                     Input Dir : ', self.inputdir)
        print ('                    Output Dir : ', self.outdir)
        print ('***************************************************\n')

        print ('********************Data MetaData****************')
        print ('            Distro Figure Save : ', self.saveFigure)
        print ('              Distro Data Save : ', self.savedistro)
        print ('            Security Evaluated : ', self.evalSec)
        print (' Security Evaluation Iteration : ', self.saveIterNumber)
        print ('***************************************************\n')

    def savehparam(self):
        if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
        tmpdir= self.outdir+'/tmp'
        if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

        outhparampath= tmpdir+'/hparams.txt'
        if not os.path.exists(outhparampath):
            number=0
            for file in os.listdir(tmpdir):
             if 'hparams' in file:
                 number+=1
            outhparampath=   tmpdir+'/hparams'+str(number)+'.txt'

        with open(outhparampath,'w') as f:
            # pickle.dump(self,f)
            values=self.__dict__
            df=pd.DataFrame([values])
            df.to_csv(f, header=vars(self), index=False)
    #
    def loadhparam(self):
        tmpdir= self.outdir+'/tmp'
        if not os.path.exists(tmpdir):
                print("no output directory detected!")
                return
        outhparampath= tmpdir+'/hparams.txt'
        if not os.path.exists(outhparampath):
            number=0
            for file in os.listdir(tmpdir):
             if 'hparams' in file:
                 number+=1
            outhparampath=   tmpdir+'/hparams'+str(number-1)+'.txt'
            print("Reading hyper parameters from : ",outhparampath)
        with open(outhparampath,'r') as f:
            df=pd.read_csv(f)
            self.layer              =   df.layer[0]
            self.hiddenode          =   df.hiddenode[0]
            self.Z_dim              =   df.Z_dim[0]
            self.X_dim              =   df.X_dim[0]
            self.Y_dim              =   df.Y_dim[0]
            self.outdir             =   df.outdir[0]
            self.inputdir           =   df.inputdir[0]
            self.minibatch          =   df.minibatch[0]
            self.chunknumber        =   df.chunknumber[0]
            self.iteration_Total    =   df.iteration_Total[0]
            self.testsamplegen      =   df.testsamplegen[0]
            self.gen_itr            =   df.gen_itr[0]
            self.dis_itr            =   df.dis_itr[0]
            self.saveFigure         =   df.saveFigure[0]
            self.savedistro         =   df.savedistro[0]
            self.evalSec            =   df.evalSec[0]
            self.saveIterNumber     =   df.saveIterNumber[0]
