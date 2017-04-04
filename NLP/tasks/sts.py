from __future__ import print_function
from __future__ import division

import numpy as np
import random
import copy
import re

from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

#from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, merge
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import backend as K

import pysts.embed
import pysts.vocab
import pysts.nlp

from . import AbstractTask

class STSTask(AbstractTask):
    def task_config(self):
        self.c['nb_runs'] = 1
        self.c['nb_epoch']   = 16#64
        self.c['patience']   = 4#16
        self.c['batch_size'] = 64#128
        self.c['num_batchs'] = 100#170
        #self.c['epoch_fract']= 1/10
        self.c['showbatchs'] = False#6
        self.c['debug']      = False
        self.c['verbose']    = 2
        #self.c['e_add_flags']= True#self.c['vocabf']
        self.c['path_embed']    = 'wordvec/'
        self.c['embdict']       = 'GloVe'#'W2V'
        self.c['embname']       = 'glove.840B'#'GoogleNews-vectors-negative%d.txt'
        self.c['embdim']        = 300
        self.c['embprune']      = 0#100
        self.c['embicase']      = True#False
        self.c['inp_e_dropout'] = 0
        self.c['inp_w_dropout'] = 0
        self.c['maskzero']  = False
        self.c['spad']      = 60
        self.c['spad0']     = self.c['spad']
        self.c['spad1']     = self.c['spad']
        #self.c['ptscorer']= None#B.mlp_ptscorer
        self.c['Ddim']    = list([1])#list([2,1])#list([])
        self.c['Dinit']   = 'he_uniform'#'glorot_uniform'#'he_normal''he_uniform'
        self.c['Dact']    = 'tanh'#'tanh'#'relu'
        self.c['Dl2reg']  = 1e-5
        self.c['fix_layers']= []
        # mainly useful for transfer learning, or 'emb' to fix embeddings
        self.c['maxscore']  = 5
        self.c['target']    = 'classes'#'classes''score1'
        self.c['opt']       = 'adam'#'sgd'#'adadelta'#'adam'
        #self.c['balance_class'] = False
        self.c['gpu'] = 1.0
    def task_initial(self):
        self.taskname = 'sts'
        self.embed= None
        self.vocab= None
        #if self.c['gpu'] != 1.0:
        #    import tensorflow as tf
        #    def get_session(gpu_fraction):
        #        print('Setting gpu_fraction to %f' %gpu_fraction)
        #        num_threads = os.environ.get('OMP_NUM_THREADS')
        #        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        #        if num_threads: return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        #        else: return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #    K.tensorflow_backend._set_session(get_session(self.c['gpu']))
    def load_sts(self, filename, skip_unlabeled=True):
        import string
        punc = string.punctuation
        punc = list("！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.・")#
        punc.extend(string.punctuation)#
        import MeCab#
        JTokenize = MeCab.Tagger("-Owakati")#chasen")
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        from nltk.stem.snowball import SnowballStemmer
        stem = SnowballStemmer('english')
        s0 = []
        s1 = []
        labels = []
        lines=open(filename,'r').read().splitlines()
        JTerror0=0
        JTerror1=0
        for i,line in enumerate(lines):
            line = line.rstrip()
            #linesplit = line.split('\t')#
            linesplit = line.split()#
            #print(linesplit)
            #print(line)
            #exit()
            if len(linesplit)==3:
                label, s0x, s1x = linesplit
            if len(linesplit)==2:
                s0x, s1x = linesplit
                label='2.5'
                print('no score')
                exit()
            if label == '':
                print('no label')
                exit()
                if skip_unlabeled: continue
                else: labels.append(-1.)
            else: labels.append(float(label))
            #s0x=word_tokenize(s0x)#
            #s1x=word_tokenize(s1x)#
            #print(s0x)
            #print(s1x)
            try:
                s0xJ=JTokenize.parse(s0x).rstrip().split(' ')#
            except:
                JTerror0+=1
                #print(i)
                #print('s0x')
                #print(line)
                #print(linesplit)
                #print(s0x)
                #print(s0xJ)
                #exit()
            try:
                s1xJ=JTokenize.parse(s1x).rstrip().split(' ')#
            except:
                JTerror1+=1
                #print(i)
                #print('s1x')
                #print(line)
                #print(linesplit)
                #print(s1x)
                #print(s1xJ)
                #exit()
            #print(s0x)
            #print(s1x)
            s0x=s0xJ
            s1x=s1xJ
            #print(s0x)
            #print(s1x)
            s0x=[word for word in s0x if word not in punc]
            s1x=[word for word in s1x if word not in punc]
            #print(s0x)
            #print(s1x)
            #print(label)
            #exit()
            #s0x=[word for word in s0x if word not in stop]
            #s1x=[word for word in s1x if word not in stop]
            #s0x=[word.lower() for word in s0x]#
            #s1x=[word.lower() for word in s1x]#
            #s0x=[stem.stem(word) for word in s0x]
            #s1x=[stem.stem(word) for word in s1x]
            s0.append(s0x)
            s1.append(s1x)
            #print(s0x)
            #print(s1x)
            #if i == 0: exit()
        #print(JTerror0)
        #print(JTerror1)
        #print(len(labels))
        #print(labels[:2])
        #print(len(s0))
        #print(s0[:2])
        #print(len(s1))
        #print(s1[:2])
        #exit()#
        return (s0, s1, np.array(labels))
    def load_vocab(self, vocabf, prune_N, icase):
        s0, s1, y  = self.load_sts(vocabf, skip_unlabeled=True)
        self.vocab = pysts.vocab.Vocabulary(s0+s1, prune_N=prune_N, icase=icase)###
    def load_embed(self, path_embed, embdict, embname, embdim):
        if embdict=='GloVe': self.embed = pysts.embed.GloVe(path_embed, glovename=embname, N=embdim)###
        if embdict=='W2V': self.embed = pysts.embed.Word2Vec(path_embed, w2vname=embname, N=embdim)###
    def task_load_resources(self, path_resources):
        self.load_embed(path_resources+self.c['path_embed'], self.c['embdict'], self.c['embname'], self.c['embdim'])
        if 'vocabf' in self.c: self.load_vocab(self.c['vocabf'], self.c['embprune'], self.c['embicase'])
    def sts_labels2categorical(self, labels, nclass):
        Y = np.zeros((len(labels), nclass))
        for j, y in enumerate(labels):
            if np.floor(y) + 1 < nclass:
                Y[j, int(np.floor(y)) + 1] = y - np.floor(y)
            Y[j, int(np.floor(y))] = np.floor(y) - y + 1
        return Y
    def task_load_data(self, path_data, filename):
        fname = path_data+self.taskname+'/'+filename
        if self.vocab is None: self.load_vocab(fname, self.c['embprune'], self.c['embicase'])
        s0, s1, y= self.load_sts(fname, skip_unlabeled=True)
        si0, sj0 = self.vocab.vectorize(s0, self.embed, spad=self.c['spad0'])###
        si1, sj1 = self.vocab.vectorize(s1, self.embed, spad=self.c['spad1'])###
        se0 = self.embed.map_jset(sj0)###
        se1 = self.embed.map_jset(sj1)###
        f0,   f1 = pysts.nlp.Sentence_flags(s0, s1, self.c['spad0'], self.c['spad1'])###
        datacleaned = {'score': y, 's0': s0, 's1': s1,
                       'score1': np.array([s/float(self.c['maxscore']) for s in y]),
                       'classes': self.sts_labels2categorical(y,int(self.c['maxscore']+1)),
                       'si0': si0, 'si1': si1, 'sj0': sj0, 'sj1': sj1, 'se0': se0, 'se1': se1, 'f0': f0, 'f1': f1}
        """print(len(s0))
        print(s0[:1])
        print(len(s1))
        print(s1[:1])
        print(len(y))
        print(y[:1])

        print(len(si0))
        print(si0[:1])
        print(len(sj0))
        print(sj0[:1])
        print(len(si1))
        print(si1[:1])
        print(len(sj1))
        print(sj1[:1])

        print(len(se0))
        print(se0[:1])
        print(len(se1))
        print(se1[:1])
        print(len(f0))
        print(f0[:1])
        print(len(f1))
        print(f1[:1])
        exit()"""
        return datacleaned

    def task_create_model(self, mod):
        K.clear_session()
        self.inputsint0 = Input(name='si0', shape=(self.c['spad0'],))
        self.inputsemb0 = Input(name='se0', shape=(self.c['spad0'], self.embed.N))
        self.inputsnlp0 = Input(name='f0',  shape=(self.c['spad0'], pysts.nlp.flagsdim))
        self.inputsint1 = Input(name='si1', shape=(self.c['spad1'],))
        self.inputsemb1 = Input(name='se1', shape=(self.c['spad1'], self.embed.N))
        self.inputsnlp1 = Input(name='f1',  shape=(self.c['spad1'], pysts.nlp.flagsdim))
        embmat = self.vocab.embmatrix(self.embed)###
        embedi = Embedding(input_dim=embmat.shape[0], input_length=self.c['spad'],
                           output_dim=self.embed.N, mask_zero=self.c['maskzero'],#True#False
                           weights=[embmat], trainable=True,
                           dropout=self.c['inp_w_dropout'], name='embedi')
        embediint0=embedi(self.inputsint0)
        embediint1=embedi(self.inputsint1)
        mergedemb0=merge([embediint0, self.inputsemb0], mode='sum', name='mergedemb0')
        mergedemb1=merge([embediint1, self.inputsemb1], mode='sum', name='mergedemb1')
        self.mergedinp0=merge([mergedemb0, self.inputsnlp0], mode='concat', name='mergedinp0')
        self.mergedinp1=merge([mergedemb1, self.inputsnlp1], mode='concat', name='mergedinp1')
        self.N = self.embed.N + pysts.nlp.flagsdim

        self.output0, self.output1 = mod.create_model(self.mergedinp0, self.mergedinp1, self.N, self.c)

        if self.c['target'] == 'classes':output_dim=int(self.c['maxscore']+1)
        if self.c['target'] == 'score1': output_dim=1
        self.outputfinal = mod.scorer(output_dim, self.output0, self.output1, self.N, self.c)

        self.model = Model(input=[self.inputsint0,self.inputsint1,self.inputsemb0,self.inputsemb1,self.inputsnlp0,self.inputsnlp1],
                           output=self.outputfinal)
        for lname in self.c['fix_layers']: self.model.nodes[lname].trainable = False
        print(self.model.layers)
        print("Constructed!")
    def task_compile_model(self):
        def pearsonobj(ny_true,ny_pred):
            my_true = K.mean(ny_true)
            my_pred = K.mean(ny_pred)
            var_true = (ny_true - my_true)**2
            var_pred = (ny_pred - my_pred)**2
            return - K.sum((ny_true - my_true) * (ny_pred - my_pred), axis=-1) / \
                     (K.sqrt(K.sum(var_true, axis=-1) * K.sum(var_pred, axis=-1)))    
        def Cpearsonobj(y_true, y_pred):
            ny_true = y_true[:,1] + 2*y_true[:,2] + 3*y_true[:,3] + 4*y_true[:,4] + 5*y_true[:,5]
            ny_pred = y_pred[:,1] + 2*y_pred[:,2] + 3*y_pred[:,3] + 4*y_pred[:,4] + 5*y_pred[:,5]
            return pearsonobj(ny_true,ny_pred)
        def Spearsonobj(y_true, y_pred):
            ny_true = y_true[:,0]
            ny_pred = y_pred[:,0]
            return pearsonobj(ny_true,ny_pred)
        if self.c['target'] == 'classes':self.c['loss']  = Cpearsonobj#'categorical_crossentropy'#O.pearsonobj
        if self.c['target'] == 'score1': self.c['loss']  = Spearsonobj#'mse'#B.pearsonobj
        self.model.compile(loss={'outputfinal': self.c['loss']}, optimizer=self.c['opt'])
        """def kl_divergence(p, q):
            cross_entropy = -tf.reduce_sum(p * tf.log(q))
            entropy = -tf.reduce_sum(p * tf.log(p + 0.00001))
            kl_div = cross_entropy - entropy
            return kl_div
        kl_loss = kl_divergence(p_, p)
        # Define regularization over all convolutional/fully connected layers
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        overal_loss= kl_loss + reg_losses"""

    def task_debug_model(self, mod):
        mod.debug_model(self.mergedinp0, self.mergedinp1, self.N, self.c, self.output0, self.output1,
                        self.inputsint0,self.inputsint1,self.inputsemb0,self.inputsemb1,self.inputsnlp0,self.inputsnlp1,self.outputfinal,
                        self.sample_pairs, self.testd)#use self.grt

    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        num = len(gr['score'])
        idN = int((num+batch_size-1) / batch_size)
        ids = list(range(num))
        while True:
            if shuffle:
                random.shuffle(ids)
            grr= copy.deepcopy(gr)
            for name, value in grr.items():
                valuer=copy.copy(value)
                for i in range(num):
                    valuer[i]=value[ids[i]]
                grr[name] = valuer
            for i in range(idN):
                sl  = slice(i*batch_size, (i+1)*batch_size)
                grsl= dict()
                for name, value in grr.items():
                    grsl[name] = value[sl]
                x = [grsl['si0'],grsl['si1'],grsl['se0'],grsl['se1'],grsl['f0'],grsl['f1']]
                y = grsl[self.c['target']]
                yield (x,y)#yield grsl
            if once: break
    def get_ypred_ygold(self, model, gr):
        batch_size = 16384#hardcoded
        ypred = []
        for grslpair,_ in self.sample_pairs(gr, batch_size, shuffle=False, once=True):
            ypred += list(model.predict(grslpair))
        if self.c['target'] == 'classes':
            ypred = np.dot(np.array(ypred),np.arange(int(self.c['maxscore']+1)))
        if self.c['target'] == 'score1':
            ypred = [y[0] for y in ypred]
            ypred = np.dot(np.array(ypred),self.c['maxscore'])
        ygold = gr['score']
        return ypred, ygold
    def task_eval_model(self, model, gr, fname, quiet=False):
        ypred, ygold = self.get_ypred_ygold(model, gr)
        STSRes = dict()
        STSRes['Pearson']  = pearsonr(ypred, ygold)[0]
        STSRes['Spearman'] = spearmanr(ypred, ygold)[0]
        STSRes['MSE']      = mse(ypred, ygold)
        if quiet: return STSRes
        print('%s Pearson: %f'  % (fname, STSRes['Pearson']))
        print('%s Spearman: %f' % (fname, STSRes['Spearman']))
        print('%s MSE: %f'      % (fname, STSRes['MSE']))
        return STSRes
    def task_pred_model(self, model, gr, fname, predout=''):
        self.task_eval_model(model, gr, fname, quiet=False)###
        ypred, ygold = self.get_ypred_ygold(model, gr)
        predname = './'+re.sub('/','',predout+'-'+fname)[1:]
        fpred=open(predname,"w")
        for ipred in range(len(ypred)):
            print("%f\t%f" %(ypred[ipred],ygold[ipred]),end="\t",file=fpred)
            print(" ".join(gr['s0'][ipred]),end="\t",file=fpred)
            print(" ".join(gr['s1'][ipred]),file=fpred)
            #print(" ".join(gr['s0'][ipred]).encode('utf-8'),end=",",file=fpred)
            #print(" ".join(gr['s1'][ipred]).encode('utf-8'),file=fpred)
        fpred.close()
    def task_fit_model(self, model, gr, grv, grt, wfname):
        kwargs = dict()
        kwargs['generator'] = self.sample_pairs(gr, self.c['batch_size'])
        kwargs['samples_per_epoch'] = self.c['num_batchs']*self.c['batch_size']#int(len(gr['score'])*self.c['epoch_fract'])
        kwargs['nb_epoch']  = self.c['nb_epoch']
        class STSPearsonCB(Callback):
            def __init__(self, task, train_gr, valid_gr, test_gr, cshow, wfname, fout):
                self.cshow  = cshow
                self.nbatch = 0
                self.best   = 0.0
                self.wfname = wfname
                self.fout   = fout
                self.task   = task
                self.train_gr = train_gr
                self.valid_gr = valid_gr
                self.test_gr  = test_gr
            def on_batch_end(self, epoch, logs={}):
                if self.cshow:
                    if self.nbatch%self.cshow==0:###
                        prvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)['Pearson']
                        if prvl > self.best:
                            self.best = prvl
                            self.task.model.save(self.wfname)
                    if self.nbatch%self.cshow==0:
                        restr = self.task.task_eval_model(self.model, self.train_gr, 'Train', quiet=True)
                        resvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)
                        rests = self.task.task_eval_model(self.model, self.test_gr,  'Test',  quiet=True)
                        print('         Pearson: train %f  valid %f  test %f' % (restr['Pearson'], resvl['Pearson'], rests['Pearson']))
                        print('        Spearman: train %f  valid %f  test %f' % (restr['Spearman'], resvl['Spearman'], rests['Spearman']))
                        print('             MSE: train %f  valid %f  test %f' % (restr['MSE'], resvl['MSE'], rests['MSE']))
                        print("%f,%f,%f" %(restr['Pearson'], resvl['Pearson'], rests['Pearson']), end="|", file=self.fout)
                        self.fout.flush()
                self.nbatch += 1
            def on_epoch_end(self, epoch, logs={}):
                if not self.cshow:
                    restr = self.task.task_eval_model(self.model, self.train_gr, 'Train', quiet=True)
                    resvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)
                    rests = self.task.task_eval_model(self.model, self.test_gr,  'Test',  quiet=True)
                    print('         Pearson: train %f  valid %f  test %f' % (restr['Pearson'], resvl['Pearson'], rests['Pearson']))
                    print('        Spearman: train %f  valid %f  test %f' % (restr['Spearman'], resvl['Spearman'], rests['Spearman']))
                    print('             MSE: train %f  valid %f  test %f' % (restr['MSE'], resvl['MSE'], rests['MSE']))
                    print("%f,%f,%f" %(restr['Pearson'], resvl['Pearson'], rests['Pearson']), end="|", file=self.fout)
                    self.fout.flush()
                    prvl = resvl['Pearson']
                    if prvl > self.best:
                        self.best = prvl
                        self.task.model.save(self.wfname)
                logs['pearson'] = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)['Pearson']
        kwargs['callbacks'] = [STSPearsonCB(self, gr, grv, grt, self.c['showbatchs'], wfname, self.c['fplot']),
            #ModelCheckpoint(wfname, save_best_only=True, monitor='pearson', mode='max'),
            EarlyStopping(monitor='pearson', mode='max', patience=self.c['patience'])]
        return model.fit_generator(verbose=self.c['verbose'],**kwargs)

def task():
    return STSTask()
