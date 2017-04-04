from __future__ import print_function
from __future__ import division

from keras.models import Model
from keras.layers import GlobalAveragePooling1D, Dense
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Activation, Dropout, merge, Merge, Flatten # merge for tensor Merge for layer
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2

from . import AbstractMod

class AVGMod(AbstractMod):
    def add_conf(self,c):
        # model-external:
        c['nb_epoch']   = 8
        c['patience']   = 8
        c['batch_size'] = 904
        c['num_batchs'] = 12#10848=32*3*113
        c['showbatchs'] = False
        #c['debug']   = True
        #c['verbose'] = 1
        #c['nb_epoch']= 1
        # dict
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
        c['nnact']  = 'relu'#'tanh''relu'
        c['nnl2reg']= 1e-5

    def create_model(self, mergedinp0, mergedinp1, N, c):
        gap1d = GlobalMaxPooling1D(name='gap1d')
        #gap1d = GlobalAveragePooling1D(name='gap1d')
        avginp0 = gap1d(mergedinp0)
        avginp1 = gap1d(mergedinp1)
        #avginp0 = Permute((2, 1))(avginp0)
        #avginp1 = Permute((2, 1))(avginp1)
        #avginp0 = TimeDistributed(Merge([mergedinp0],mode='ave'))#(mergedinp0)
        #avginp1 = TimeDistributed(Merge([mergedinp1],mode='ave'))#(mergedinp1)
        output0 = avginp0
        output1 = avginp1
        dense = []
        for i in range(c['nndeep']):
            dense.append(Dense(name='nndeep%d'%(i,), output_dim=N, activation=c['nnact'], init=c['nninit'], W_regularizer=l2(c['nnl2reg'])))
        for i in range(c['nndeep']):
            output0 = dense[i](output0)
            output1 = dense[i](output1)
        self.testlayers.append(avginp0)
        self.testlayers.append(output0)
        return output0, output1

def mod():
    return AVGMod()
