from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict

def default_config():
    c = OrderedDict()
    c['nb_runs'] = 1
    c['nb_epoch']   = 16
    c['patience']   = 4
    c['batch_size'] = 64
    c['num_batchs'] = 50
    c['epoch_fract']= 1
    return c

class AbstractTask(object):
    def __init__(self):
        self.c = default_config()
        self.task_config()
    def get_conf(self):
        return self.c
    def set_conf(self, c):
        self.c = c
    def initial(self):
        self.task_initial()
    def load_resources(self, path_resources):
        return self.task_load_resources(path_resources)
    def load_data(self, path_data, trainf, validf, testf):
        self.trainf= trainf
        self.validf= validf
        self.testf = testf
        self.traind= self.task_load_data(path_data,trainf)
        self.validd= self.task_load_data(path_data,validf)
        self.testd = self.task_load_data(path_data,testf)

    def get_model(self):
        return self.model
    def set_model(self, model):
        self.model = model

    def create_model(self, mod):
        self.task_create_model(mod)
    def compile_model(self):
        self.task_compile_model()
    def debug_model(self, mod):
        self.task_debug_model(mod)

    def eval_model(self):
        res = []
        for datad, fname in [(self.traind, self.trainf), (self.validd, self.validf), (self.testd, self.testf)]:
            res.append(self.task_eval_model(self.model, datad, fname))
        return tuple(res)
    def pred_model(self, predout):
        for datad, fname in [(self.traind, self.trainf), (self.validd, self.validf), (self.testd, self.testf)]:
            self.task_pred_model(self.model, datad, fname, predout=predout)
    def fit_model(self, wfname):
        return self.task_fit_model(self.model, self.traind, self.validd, self.testd, wfname)

    #def purepredict(self, path_data, predin, predout):
    #    predinfiles=predin.split(',')
    #    for predinfile in predinfiles:
    #        self.task_purepredict(self.model, path_data, predinfile, predout)

