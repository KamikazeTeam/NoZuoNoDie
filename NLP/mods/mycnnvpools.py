from __future__ import print_function
from __future__ import division

from keras.models import Model
from keras.layers import GlobalAveragePooling1D, Dense
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers import Activation, Dropout, merge, Merge, Flatten # merge for tensor Merge for layer
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2

from . import AbstractMod

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def _conv_bn_relu(nb_filter, filter_length, subsample_length=1):
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, subsample_length=subsample_length,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)
    return f

def _bn_relu_conv(nb_filter, filter_length, subsample_length=1):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution1D(nb_filter=nb_filter, filter_length=filter_length, subsample_length=subsample_length,
                             init="he_normal", border_mode="same")(activation)
    return f

def _basic_block(nb_filter, filter_length, subsample_length=1):
    def f(input):
        conv1 = _bn_relu_conv(nb_filter, filter_length, subsample_length)(input)
        residual = _bn_relu_conv(nb_filter, filter_length, 1)(conv1)
        if subsample_length == 1:
            shortcut = input
        else:
            shortcut = Convolution1D(nb_filter=residual._keras_shape[1], filter_length=filter_length, subsample_length=subsample_length,
                                     init="he_normal", border_mode="valid")(input)
        return merge([shortcut, residual], mode="sum")
    return f

def _residual_block(block_function, nb_filter, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            subsample_length = 1
            if i == 0 and not is_first_layer: subsample_length = 2
            input = block_function(nb_filter=nb_filter, filter_length=filter_length, subsample_length=subsample_length)(input)
        return input
    return f

def _resnet():
    def f(input):
        conv1 = _basic_block(nb_filter=304, filter_length=1, subsample_length=1)(input)
        pool1 = MaxPooling1D(pool_length=30, stride=1, border_mode="same")(conv1)

        # Build residual blocks..
        #block_fn = _basic_block
        #block1 = _residual_block(block_fn, nb_filter=64, repetations=3, is_first_layer=True)(pool1)
        #block2 = _residual_block(block_fn, nb_filter=128, repetations=4)(block1)
        #block3 = _residual_block(block_fn, nb_filter=256, repetations=6)(block2)
        #block4 = _residual_block(block_fn, nb_filter=512, repetations=3)(block3)

        # Classifier block
        pool2 = pool1#MaxPooling1D(pool_length=(7, 7), strides=(1, 1), border_mode="same")(block4)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=300, init="he_normal", activation="softmax")(flatten1)
        return dense
    return f

class CNNMod(AbstractMod):
    def add_conf(self,c):
        # model-external:
        c['nb_epoch']   = 6#8
        c['patience']   = 6#8
        c['batch_size'] = 339#904
        c['num_batchs'] = 32#12#10848=32*3*113#11901,1377
        c['showbatchs'] = 8#False
        #c['debug']   = True
        #c['verbose'] = 1
        #c['nb_epoch']= 1
        # dict
        #c['embdict']       = 'W2V'
        #c['embname']       = 'GoogleNews-vectors-negative%d.txt'#'GoogleNews-vectors-negative%d.bin.gz'
        c['embdict']       = 'GloVe'
        c['embname']       = 'glove.840B'
        c['embdim']        = 300
        c['embprune']      = 0
        c['embicase']      = True
        c['inp_e_dropout'] = 0
        c['inp_w_dropout'] = 0
        c['maskzero']  = False
        c['spad']      = 30#60
        c['spad0']     = c['spad']
        c['spad1']     = c['spad']
        # mlp
        c['Ddim']    = list([1])
        c['Dinit']   = 'he_uniform'
        c['Dact']    = 'tanh'
        c['Dl2reg']  = 1e-5
        # dense dim=N
        c['nndeep'] = 0
        c['nninit'] = 'he_uniform'#'glorot_uniform'#'he_normal''he_uniform'
        c['nnact']  = 'relu'#'tanh'
        c['nnl2reg']= 1e-5
        # cnn:
        c['cdim']       = {1: 1}#, 2: 1/2}#, 3: 1/2, 4: 1/2}
        c['cinit']      = 'he_uniform'
        c['cact']       = 'relu'
        c['cl2reg']     = 1e-5#4
        c['cdropout']   = 0

    def create_model(self, mergedinp0, mergedinp1, N, c):
        if 0:
            resnet  = _resnet()
            merged0 = resnet(mergedinp0)
            merged1 = resnet(mergedinp1)
            return merged0, merged1
        con,bat,act,dpt=[],[],[],[]
        co2,ba2,ac2,dp2=[],[],[],[]
        gmp,amp,flt=[],[],[]
        for fl, cd in c['cdim'].items():
            nb_filter = int(N*cd)
            con.append(Convolution1D(name='con%d'%(fl,), nb_filter=nb_filter, filter_length=fl, border_mode='valid',#valid same
                                     init=c['cinit'], W_regularizer=l2(c['cl2reg'])))#subsample_length=1))
            bat.append(BatchNormalization(mode=2, axis=1))
            act.append(Activation(c['cact']))
            dpt.append(Dropout(name='dpt%d'%(fl,),p=c['cdropout']))
            co2.append(Convolution1D(name='co2%d'%(fl,), nb_filter=nb_filter, filter_length=fl, border_mode='valid',#valid same
                                     init=c['cinit'], W_regularizer=l2(c['cl2reg'])))#subsample_length=1))
            ba2.append(BatchNormalization(mode=2, axis=1))
            ac2.append(Activation(c['cact']))
            dp2.append(Dropout(name='dp2%d'%(fl,),p=c['cdropout']))
            gmp.append(MaxPooling1D(name='gmp%d'%(fl,), pool_length=int(c['spad']-fl+1-fl+1)))#-fl+1
            amp.append(AveragePooling1D(name='amp%d'%(fl,), pool_length=int(c['spad']-fl+1-fl+1)))#-fl+1
            flt.append(Flatten(name='flt%d'%(fl,)))
        def mergego(input00, input01, input10, input11, mode, flag):
            if not flag: return input00, input01
            output0,output1=[],[]
            for fl in range(len(c['cdim'])):
                output0.append(merge([input00[fl], input10[fl]], mode=mode))
                output1.append(merge([input01[fl], input11[fl]], mode=mode))
            return output0, output1
        def layergo(input0, input1, layer, flag):
            if not flag: return input0, input1
            output0,output1=[],[]
            for fl in range(len(c['cdim'])):
                output0.append(layer[fl](input0[fl]))
                output1.append(layer[fl](input1[fl]))
            return output0, output1
        input0=[]
        input1=[]
        for fl in range(len(c['cdim'])):
            input0.append(mergedinp0)
            input1.append(mergedinp1)
        convol0, convol1=layergo( input0, input1, con, 1)
        batnor0, batnor1=layergo(convol0,convol1, bat, 0)
        output0, output1=batnor0, batnor1
        if 0:
            ac2ive0, ac2ive1=layergo(batnor0,batnor1, ac2, 1)
            dr2out0, dr2out1=layergo(ac2ive0,ac2ive1, dp2, 0)
            co2vol0, co2vol1=layergo(dr2out0,dr2out1, co2, 1)
            ba2nor0, ba2nor1=layergo(co2vol0,co2vol1, ba2, 0)
            resnet0, resnet1=mergego(ba2nor0,ba2nor1, input0, input1, 'sum', 0)
            output0, output1=resnet0, resnet1
        active0, active1=layergo(output0,output1, act, 1)
        drpout0, drpout1=layergo(active0,active1, dpt, 0)
        gmpool0, gmpool1=layergo(drpout0,drpout1, gmp, 1)
        ampool0, ampool1=layergo(drpout0,drpout1, amp, 1)
        vpools0, vpools1=mergego(gmpool0,gmpool1, ampool0, ampool1, 'concat', 1)
        flated0, flated1=layergo(vpools0,vpools1, flt, 1)
        if len(c['cdim']) > 1:
            merged0=merge(flated0,name='merged0',mode='concat')
            merged1=merge(flated1,name='merged1',mode='concat')
        else:
            merged0=flated0[0]
            merged1=flated1[0]
        dense = []
        for i in range(c['nndeep']):
            dense.append(Dense(name='nndeep%d'%(i,), output_dim=N, activation=c['nnact'], init=c['nninit'], W_regularizer=l2(c['nnl2reg'])))
        for i in range(c['nndeep']):
            merged0 = dense[i](merged0)
            merged1 = dense[i](merged1)
        self.testlayers.append(convol0[0])
        self.testlayers.append(drpout0[0])
        self.testlayers.append(gmpool0[0])
        self.testlayers.append(flated0[0])
        self.testlayers.append(merged0)
        return merged0, merged1
        #Activation('linear') Dropout

def mod():
    return CNNMod()
