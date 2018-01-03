# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:10:43 2017
GAN-Sec: Conditional Generative Adversarial Network Modeling for security
analysis of Cyber-Physical Production Systems
@author: Sujit0, Anthony
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
from sklearn import preprocessing
import math
import random
from sklearn.neighbors.kde import KernelDensity
import argparse
import time
import sys
from hparams import hyperparameters

# some variables for data management

plt.switch_backend('agg')
min_max_scaler  = preprocessing.MinMaxScaler()
#%%
def parsingInit():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cn","--chunknumber", type=int, nargs='?',
                        default=900,
                        help="Total chunks to take from each folder")
    parser.add_argument("-itr","--iterationnumber", type=int, nargs='?',
                        default=2000,
                        help="Total iteration")
    parser.add_argument("-tgs","--testgensample", type=int, nargs='?',
                        default=2000,
                        help="Samples for distribution generation")
    parser.add_argument("-z","--zsize", type=int, nargs='?',
                        default=100,
                        help="size of batch size for noise")
    parser.add_argument("-dtn","--disctrainnumber", type=int, nargs='?',
                        default=2,
                        help="training number for discriminator")
    parser.add_argument("-gtn","--gentrainnumber", type=int, nargs='?',
                        default=2,
                        help="training number for discriminator")
    parser.add_argument("-node","--hiddennodes", type=int, nargs='?',
                        default=128,
                        help="hidden Nodes")
    parser.add_argument("-layer","--layer", type=int, nargs='?',
                        default=2,
                        help="batch size")
    parser.add_argument("-bs","--batchsize", type=int, nargs='?',
                        default=64,
                        help="batch size")
    parser.add_argument("-sf","--savefigure", type=bool, nargs='?',
                        default=True,
                        help="saves the figures")
    parser.add_argument("-sd","--savedistro", type=bool, nargs='?',
                        default=True,
                        help="saves the distro data")
    parser.add_argument("-seceval","--securityeval", type=bool, nargs='?',
                        default=True,
                        help="evaluate Security")
    parser.add_argument("-out","--outputfile", type=str, nargs='?',
                       default="../out",
                       help="Output file directory")
    parser.add_argument("-saveitr","--saveiteration", type=int, nargs='?',
                      default=50,
                      help="saving security evaluation every")
    parser.add_argument("-xdim","--inputdimension", type=int, nargs='?',
                      default=100,
                      help="Input dimension for discriminator")
    parser.add_argument("-ydim","--conditiondimension", type=int, nargs='?',
                     default=4,
                     help="Condition dimension for discriminator/Generator")
    parser.add_argument("-inputdir","--inputdirectory", type=str, nargs='?',
                    default="../data",
                    help="Input data directory")

    args = parser.parse_args()

    return  (args.chunknumber,
             args.iterationnumber,
             args.testgensample,
             args.zsize,
             args.disctrainnumber,
             args.hiddennodes,
             args.batchsize,
             args.savefigure,
             args.layer,
             args.gentrainnumber,
             args.savedistro,
             args.securityeval,
             args.outputfile,
             args.saveiteration,
             args.inputdimension,
             args.conditiondimension,
             args.inputdirectory
             )


(cn, Iter, testgen, NOISE_SIZE,
D_TRAIN_ITERS,h_dim,BATCH_SIZE,
sf,layer,gentrain, savedistro,
securityeval,outdir,saveiteration, X_dim,y_dim,inputdir) = parsingInit()

hparam=hyperparameters(layer=layer, hiddenode=h_dim, Z_dim=NOISE_SIZE,
X_dim=X_dim, Y_dim=y_dim, outdir=outdir, inputdir=inputdir,
minibatch=BATCH_SIZE, chunknumber=cn,
iteration_Total=Iter, testsamplegen=testgen,
gen_itr=gentrain, dis_itr=D_TRAIN_ITERS, saveFigure=sf,
savedistro=savedistro, evalSec=securityeval,saveIterNumber=saveiteration)

mb_size         = hparam.minibatch  #batch size
Z_dim           = hparam.Z_dim # noise vector length (input to generator)
X_dim           = hparam.X_dim # place holder ; input size - emissions
y_dim           = hparam.Y_dim #place holder ; label size - component
TEST_SIZE       = hparam.minibatch
OUTPUT_DIR      = outdir+"/layer_"+str(hparam.layer)+ \
                  "_hidden_"+str(hparam.hiddenode)+"_destrain_"+ \
                  str(hparam.dis_itr)+"_gentrain_"+ \
                   str(hparam.gen_itr)+ "_iter_"+str(hparam.iteration_Total)+\
                   "_tgs_"+str(hparam.testsamplegen)

chunknumberFlag = False
hparam.outdir= OUTPUT_DIR

if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#%%
def format_data(chunknumber):
    global chunknumberFlag
    # read the data from the specified file path name
    filepath=hparam.inputdir+'/chunk'+str(chunknumber)+'.csv'
    while not os.path.exists(filepath):
            chunknumber=chunknumber+1
            filepath=hparam.inputdir+'/chunk'+str(chunknumber)+'.csv'
            if chunknumber==hparam.chunknumber:
                return
    with open(filepath) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            x = list(csvReader)
            data = np.array(x).astype("float32")
    # TODO: SHUFFLE CHUNKS
    if not chunknumberFlag:
        val=random.randrange(0, hparam.chunknumber, 1)
        filepath_rnd=hparam.inputdir+'/chunk'+str(val)+'.csv'

        with open(filepath_rnd) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                x = list(csvReader)
                data = np.array(x).astype("float32")
        rand_data_minmax = min_max_scaler.fit_transform(data[:,1:1+hparam.X_dim])
        chunknumberFlag=True
    else:
        rand_data_minmax = min_max_scaler.transform(data[:,1:1+hparam.X_dim])

    return (np.concatenate((rand_data_minmax, data[:, 1+hparam.X_dim:1+hparam.X_dim+hparam.Y_dim]),axis=1))

#%%
def get_batch(data,curr_row):
    x_mb = data[curr_row:curr_row+hparam.minibatch, 0:hparam.X_dim]
    y_mb = data[curr_row:curr_row+hparam.minibatch, hparam.X_dim:hparam.X_dim+hparam.Y_dim]
    # return the batch of samples
    return x_mb, y_mb
#%%
if hparam.layer==2:
   # Discriminator Net model
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])

    D_W1 = tf.Variable(xavier_init([X_dim + y_dim, hparam.hiddenode]))
    D_b1 = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))

    D_W2 = tf.Variable(xavier_init([hparam.hiddenode, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    #%% TODO: Change relu layer to leaky relu layer
    def discriminator(x, y):
        inputs = tf.concat(axis=1, values=[x, y])
        D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    #%% Generator Net model
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, hparam.hiddenode]))
    G_b1 = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))

    G_W2 = tf.Variable(xavier_init([hparam.hiddenode, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]
    #%%
    # TODO: Normalize output of generator
    # TODO: Change relu hparam.layer to leaky relu layer
    def generator(z, y):
        inputs = tf.concat(axis=1, values=[z, y])
        G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

elif hparam.layer==3:
   # Discriminator Net model
    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])

    D_W1 = tf.Variable(xavier_init([X_dim + y_dim, hparam.hiddenode]))
    D_b1 = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))
    D_Wtest = tf.Variable(xavier_init([hparam.hiddenode, hparam.hiddenode]))
    D_btest = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))
    D_W2 = tf.Variable(xavier_init([hparam.hiddenode, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_Wtest, D_W2, D_b1, D_btest, D_b2]

    #%% TODO: Hyperparameter training for layers
    def discriminator(x, y):
        inputs = tf.concat(axis=1, values=[x, y])
        D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
        D_htest = tf.nn.leaky_relu(tf.matmul(D_h1, D_Wtest) + D_btest)
        D_logit = tf.matmul(D_htest, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    #%% Generator Net model
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
    # TODO: INITIALIZE BIAS
    G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, hparam.hiddenode]))
    G_b1 = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))
    G_Wtest = tf.Variable(xavier_init([hparam.hiddenode, hparam.hiddenode]))
    G_btest = tf.Variable(tf.zeros(shape=[hparam.hiddenode]))
    G_W2 = tf.Variable(xavier_init([hparam.hiddenode, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_Wtest, G_W2, G_b1, G_btest, G_b2]

    #%%
    # TODO: Hyperparameter training for layers
    def generator(z, y):
        inputs = tf.concat(axis=1, values=[z, y])
        G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1)
        G_htest = tf.nn.leaky_relu(tf.matmul(G_h1, G_Wtest) + G_btest)
        G_log_prob = tf.matmul(G_htest, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

else:
    sys.exit("No support for higher layer than 3 now")

#%%
def sample_Z(m, n):
    stddev = 1. / math.sqrt(m / 2.)
    return np.random.normal(0, stddev, [m,n])

#%% uses the discriminator to evaluate the liklihood of all the given features (row by row)
def discr_metric(test_data, cond_label, sess, D_real, D_logit_real):
    row = 0
    disc_metric_correct = 0
    disc_metric_incorrect = 0

    num_correct = 0
    num_incorrect = 0
    for iters in range(int(len(test_data)/TEST_SIZE)): # TEST SIZE ITERS
        # discriminator metric

        curr_data = test_data[row:row+TEST_SIZE, 0:hparam.X_dim]
        true_label = test_data[row:row+TEST_SIZE, hparam.X_dim:hparam.X_dim+4] #get the true sample labels
        for i in range(TEST_SIZE):
            # get the probability that this belongs to the real data according to the discriminator
            probs, _ = sess.run([D_real, D_logit_real], feed_dict={X: np.reshape(curr_data[i], (1,hparam.X_dim)), y: np.reshape(true_label[i], (1,hparam.Y_dim))})
            #print("Distr probs " + str(probs))
            # update the metric according to the probability (signs could be changed according to perspective)
            if(true_label[i, cond_label] == 1): # this is positive weight e.g. attack success
                #for i in range(len(probs)):
                disc_metric_correct += probs
                num_correct += 1
            else: # this is negative weight, e.g., attack failure
                #for i in range(len(probs)):
                disc_metric_incorrect += probs
                num_incorrect += 1
        if((iters*TEST_SIZE) + TEST_SIZE > len(test_data)):
            break; # break out of the loop and return
        else:
            row += TEST_SIZE
    disc_metric_correct = disc_metric_correct / num_correct
    disc_metric_incorrect = disc_metric_incorrect / num_incorrect
    return disc_metric_correct, disc_metric_incorrect

#%% evaluate the overall security metric over all the test data for the given feature columns
def security_metric(test_data, G_sample, H, kde, feat_cols, cond_label):
    row = 0
    gen_metric_correct = 0 # generator metric
    gen_metric_incorrect = 0
    num_correct = 0
    num_incorrect = 0
    # for each TEST_SIZE of test data
    for iters in range(int(len(test_data)/TEST_SIZE)): # TEST SIZE ITERS
        #curr_data = test_data(row:row+TEST_SIZE, 0:hparam.X_dim) # get the sample values
        curr_data = test_data[row:row+TEST_SIZE, feat_cols]
        true_label = test_data[row:row+TEST_SIZE, hparam.X_dim:hparam.X_dim+4] #get the true sample labels
        # compute our metrics from the conditional probabilities for each sample
        # go through each sample

        for i in range(TEST_SIZE):
            # for each sample, evaluate the probability using the kde of each conditional distribution
            log_probs = kde.score_samples(curr_data[i])
            # since the returned values are log probabilities, take the exponential
            # multiply the window size to get actual probability
            probs = np.exp(log_probs) * H
            #print("Gener probs: " + str(probs))
            # update the metric according to the probability (signs could be changed according to perspective)
            if(true_label[i, cond_label] == 1): # this is positive weight e.g. attack success
                #for i in range(len(probs)):
                gen_metric_correct += probs
                num_correct += 1
            else: # this is negative weight, e.g., attack failure
                #for i in range(len(probs)):
                gen_metric_incorrect += probs
                num_incorrect += 1
        if((iters*TEST_SIZE) + TEST_SIZE > len(test_data)):
            break; # break out of the loop and return
        else:
            row += TEST_SIZE
    # averaged overall security metric (for c, i, a) over all of test samples
    gen_metric_correct = gen_metric_correct /num_correct
    gen_metric_incorrect = gen_metric_incorrect / num_incorrect
    return gen_metric_correct, gen_metric_incorrect

#%% we will use this function to initialize the discriminator before training the entire CGAN
def build_model():
    print("Building and initializing the generator and discriminator")
    for j in range(hparam.dis_itr):
        G_sample = generator(Z, y)
        D_real, D_logit_real = discriminator(X, y)
        D_fake, D_logit_fake = discriminator(G_sample, y)

        # this section defines the loss functions
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        # TODO: change to max log D instead of min 1-log (D)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    return (D_loss, G_loss, G_sample, D_real, D_logit_real)

#%%### HERE IS THE MAIN CODE ####
def model_INIT():
     # format the data
    (D_loss, G_loss, G_sample, D_real, D_logit_real) = build_model() # pretrain the discriminator and the generator


    # this section defines training methods according to the loss functions
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # the CGAN training officially starts here
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth = True)
    sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,  allow_soft_placement=True, log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    return sess,  D_solver, G_solver, D_loss, G_loss, G_sample, D_real, D_logit_real
#%%
def saveFigure(sess,
               LABEL_SIZE,
               test_gen_sample,
               Z_dim,
               y_dim,G_sample,currentChunk):
    H_ITERS = 5
    for cond_label in range(hparam.Y_dim-1): #excluding E motor for now
        n_sample = test_gen_sample
        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, cond_label] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        X_plot = np.linspace(-1.1, 1.1, n_sample)[:, np.newaxis]
        distrodir=OUTPUT_DIR+'/distro_figures'

        if not os.path.exists(distrodir):
                os.makedirs(distrodir)

        for features in range(np.shape(samples)[1]):
            fig, ax = plt.subplots()
            for H in (np.arange(H_ITERS)+1):
                X_sample=np.reshape(samples[:,features],(n_sample,1))
                kde = KernelDensity(kernel='gaussian', bandwidth=H/H_ITERS).fit(X_sample) #H/H_ITERS
                distr = np.exp(kde.score_samples(X_plot))
                ax.plot(X_plot[:, 0], distr, '-', label="kernel = Gaussian, \
                    label: 'H = '{}'".format(np.array(H/H_ITERS)))

                ax.text(6, 0.38, "N={0} points".format(n_sample))
                ax.legend(loc='upper left')
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-.2, 4)

            fig.savefig(distrodir+'/Distribution_Feature_'+str(features)+'_Chunk_'+str(currentChunk)+'_Condition_'+str(cond_label)+'.png')
            plt.close(fig)
#%%
def saveDistroData(kde,features,cond_label,H,n_sample,currentChunk):
    X_plot = np.linspace(-1.1, 1.1, n_sample)[:, np.newaxis]
    distr = np.exp(kde.score_samples(X_plot))
    df=pd.DataFrame([distr])
    frame=[pd.DataFrame([currentChunk]), df]
    df=pd.concat(frame,axis=1)

    distrodir=OUTPUT_DIR+'/distro_data'

    if not os.path.exists(distrodir):
            os.makedirs(distrodir)

    with open(distrodir+'/Distribution_Feature_'+str(features)+'_Condition_'+str(cond_label)+'_H_'+str(H)+'.csv','a') as f:
        df.to_csv(f, header=False, index=False, sep=',')

#%%
def securityeval(sess,G_sample,D_real, D_logit_real,currentChunk,test_gen_sample,cn):
    H_ITERS = 5
    scoredir=OUTPUT_DIR+'/scores'

    if not os.path.exists(scoredir):
            os.makedirs(scoredir)

    for cond_label in range(hparam.Y_dim-1): #excluding E motor for now
        n_sample = test_gen_sample
        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, cond_label] = 1
        # generate samples from Generator
        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})
        if((currentChunk+1) <= hparam.chunknumber ):
            test_data = format_data(currentChunk+1)
            # Test our discriminator metric
            if hparam.evalSec:
                disc_sec_metric_c, disc_sec_metric_ic = discr_metric(test_data, cond_label, sess, D_real, D_logit_real)
            # generate distribution for each feature and for each parzen window H


            for H in (np.arange(H_ITERS)+1):
                if hparam.evalSec:
                    df_c=pd.DataFrame()
                    df_ic = pd.DataFrame()
                for features in range(np.shape(samples)[1]):
                    X_sample=np.reshape(samples[:,features],(n_sample,1))
                    #print(np.shape(X_sample))
                    kde = KernelDensity(kernel='gaussian', bandwidth=H/H_ITERS).fit(X_sample) #H/H_ITERS
                    #Store the distribution data
                    if hparam.savedistro:
                        saveDistroData(kde,features,cond_label,H/H_ITERS,n_sample,currentChunk)
                    feat_cols = features
                    if hparam.evalSec:
                        f_sec_metric_c, f_sec_metric_ic = security_metric(test_data, G_sample, H/H_ITERS, kde, feat_cols, cond_label)
#                    if(features == 50 and (H/H_ITERS) == .2):
#                        print("Discrim Probs: " + str(disc_security_metric) + ", Gen Metric: " + str(feature_security_metric) + ", Curr Feature: " + str(feat_cols) + ", Curr Label: " + str(cond_label) )
                        if features==0:
                            df_c=pd.DataFrame(f_sec_metric_c)
                            df_ic = pd.DataFrame(f_sec_metric_ic)
                        else:
                            df1_c=pd.DataFrame(f_sec_metric_c)
                            df1_ic = pd.DataFrame(f_sec_metric_ic)
                            frame_c=[df_c, df1_c]
                            frame_ic=[df_ic, df1_ic]
                            df_c=pd.concat(frame_c, axis=1)
                            df_ic=pd.concat(frame_ic, axis=1)
                if hparam.evalSec:
                    frame_c=[pd.DataFrame([currentChunk]),pd.DataFrame(disc_sec_metric_c),df_c]
                    df_c=pd.concat(frame_c,axis=1)
                    frame_ic = [pd.DataFrame([currentChunk]),pd.DataFrame(disc_sec_metric_ic),df_ic]
                    df_ic = pd.concat(frame_ic,axis=1)
                    with open(scoredir+'/Score_Condition_'+str(cond_label)+'_True_Positive_H_'+str(H)+'.csv','a') as f:
                        df_c.to_csv(f, header=False, index=False, sep=',')
                    with open(scoredir+'/Score_Condition_'+str(cond_label)+'_False_Positive_H_'+str(H)+'.csv','a') as f:
                        df_ic.to_csv(f, header=False, index=False, sep=',')

#%%
def train(sess, D_solver, G_solver, D_loss,  G_loss,
          G_sample, D_real, D_logit_real):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    iteration_Total=0
    perm=np.arange(hparam.chunknumber)
    np.random.shuffle(perm)
    currentchunkcount=0
    for currentChunk in perm:
        print("Chunk Number:", currentchunkcount)

        filepatht=hparam.inputdir+'/chunk'+str(currentChunk)+'.csv'
        if not os.path.exists(filepatht):
            print("Chunk Number:", currentChunk, "doesnot exists! moving on...")
            continue
        start_time = time.time()
        data_matrix = format_data(currentChunk)

        curr_row=0

        for it in range(hparam.iteration_Total ):

            if(curr_row+hparam.minibatch < len(data_matrix)-1):
                X_mb, y_mb = get_batch(data_matrix,curr_row) # get next BATCH_SIZE samples
                Z_sample = sample_Z(mb_size, Z_dim)

                for dtitr in range (hparam.dis_itr):
                    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
                for gitr in range (hparam.gen_itr):
                    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

                if it % 50 == 0:
                    df=pd.DataFrame([[currentChunk,iteration_Total,D_loss_curr,G_loss_curr]])
                    with open(OUTPUT_DIR+'/traineval.csv','a') as f:
                        df.to_csv(f, header=False, index=False)


                if it % 500 == 0:
                     print('Iter: {}'.format(iteration_Total))
                     print('D loss: {:.4}'.format(D_loss_curr))
                     print('G_loss: {:.4}'.format(G_loss_curr))
                     print()
                curr_row += hparam.minibatch

            else:
                 curr_row=0
            iteration_Total+=1

        print("Training Time for Chunk", currentchunkcount,
                     ":", (time.time() - start_time), " seconds!")

        if currentchunkcount % 100 ==0:
            if hparam.saveFigure==True:
                saveFigure(sess, hparam.Y_dim,hparam.testsamplegen, Z_dim, y_dim,G_sample,currentChunk)

        if currentchunkcount % hparam.saveIterNumber == 0:
            start_time = time.time()
            securityeval(sess,G_sample,D_real, D_logit_real,currentChunk,hparam.testsamplegen,hparam.chunknumber)
            print("Security/Distro Evulation Time :", (time.time() - start_time), " seconds!")

        currentchunkcount+=1

    sess.close()
    return

#%%
def main(argv=None):

     #initialize the model
    (sess,  D_solver, G_solver, D_loss,
        G_loss, G_sample, D_real, D_logit_real)= model_INIT()

    hparam.printhparam()
    hparam.savehparam()

    # call the training fuction
    train(sess, D_solver, G_solver, D_loss,  G_loss,
          G_sample, D_real, D_logit_real)


#%%
if __name__=="__main__":
    tf.app.run()
