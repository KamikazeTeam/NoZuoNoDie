"""Skip-thoughts use unigram/bigram information from the Children Book dataset."""
from __future__ import print_function
import numpy as np
import os
try:
    import skipthoughts
    skipthoughts_available = True
except ImportError:
    skipthoughts_available = False

class Embedder(object):
    """* w: dict mapping tokens to indices
       * g: matrix with one row per token index
       * N: embedding dimensionality"""
    def map_tokens(self, tokens, ndim=2):
        gtokens = [self.g[self.w[t]] for t in tokens if t in self.w]
        if not gtokens: return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)###
        gtokens = np.array(gtokens)
        if ndim == 2: return gtokens
        else: return gtokens.mean(axis=0)
    def map_set(self, ss, ndim=2):
        return [self.map_tokens(s, ndim=ndim) for s in ss]
    def map_jset(self, sj):
        return self.g[sj]

    def pad_set(self, ss, spad, N=None):
        ss2 = []
        if N is None: N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2: s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else: s = np.hstack((s, np.zeros(spad - s.shape[0])))  # pad non-embeddings (e.g. toklabels) too
            elif spad < s.shape[0]: s = s[:spad]
            ss2.append(s)
        return np.array(ss2)

from nltk.stem.snowball import SnowballStemmer
stem = SnowballStemmer('english')
import multiprocessing as mp
import re
def worker(para):
    w = dict()
    g = []
    for line in para[1]:#lines
        ###l2 = re.split('\t| ',line.strip())
        l = line.split(' ')
        #l[0]=stem.stem(l[0])
        w[l[0]] = len(g)+para[0]+1#word#block
        g.append(np.array(l[1:]).astype(float))
    return (w,g)
class GloVe(Embedder):
    def __init__(self, path_embed, glovename, N):
        """self.N = N
        self.w = dict()
        self.g = []
        self.glovepath = glovepath % (N,)

        # [0] must be a zero vector
        self.g.append(np.zeros(self.N))

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split(' ')
                word = l[0]
                self.w[word] = len(self.g)
                self.g.append(np.array(l[1:]).astype(float))
        self.g = np.array(self.g, dtype='float32')"""
        print('Load GloVe')
        self.N = N
        self.w = dict()
        self.g = []
        self.g.append(np.zeros(self.N))
        self.glovename = path_embed + glovename + '.%d'%(N,)
        lines = open(self.glovename, 'r').readlines()
        numbs = 1000
        r_list = mp.Pool(32).map(worker, ((blk,lines[blk:blk+numbs]) for blk in range(0,len(lines),numbs)))
        for r in r_list:
          self.w.update(r[0])#rw
          self.g.extend(r[1])#rg
        self.g = np.array(self.g, dtype='float32')

class Word2Vec(Embedder):
    def __init__(self, path_embed, w2vname, N):
        print('Load W2V')
        self.N = N
        self.w = dict()
        self.g = []
        self.g.append(np.zeros(self.N))
        self.w2vname = path_embed + w2vname % (N,)
        """lines = open(self.w2vname, 'r').readlines()
        lines = lines[1:]
        numbs = 1000
        r_list = mp.Pool(32).map(worker, ((blk,lines[blk:blk+numbs]) for blk in range(0,len(lines),numbs)))
        for r in r_list:
          self.w.update(r[0])#rw
          self.g.extend(r[1])#rg
        self.g = np.array(self.g, dtype='float32')"""
        """import gensim
        gdict = gensim.models.Word2Vec.load_word2vec_format(self.w2vname, binary=True)
        #assert self.N == self.g.vector_size
        for tok in gdict:
            self.w[tok] = len(self.g)
            self.g.append(np.array(gdict[tok]).astype(float))
        self.g = np.array(self.g, dtype='float32')"""

class SkipThought(Embedder):
    def __init__(self, datadir, uni_bi="combined"):
        """ Embed Skip_Thought vectors, using precomputed model in npy format.
        Args: uni_bi: possible values are "uni", "bi" or "combined" determining what kind of embedding should be used.
        todo: is argument ndim working properly?"""
        import skipthoughts
        self.encode = skipthoughts.encode

        if datadir is None: datadir = os.path.realpath('__file__')
        self.datadir = self.datadir

        # table for memoizing embeddings
        self.cache_table = {}

        self.uni_bi = uni_bi
        if uni_bi in ("uni", "bi"): self.N = 2400
        elif uni_bi == "combined": self.N = 4800
        else: raise ValueError("uni_bi has invalid value. Valid values: 'uni', 'bi', 'combined'")

        self.skipthoughts.path_to_models = self.datadir
        self.skipthoughts.path_to_tables = self.datadir
        self.skipthoughts.path_to_umodel = skipthoughts.path_to_models + 'uni_skip.npz'
        self.skipthoughts.path_to_bmodel = skipthoughts.path_to_models + 'bi_skip.npz'
        self.st = skipthoughts.load_model()

    def map_tokens(self, tokens, ndim=2):
        """Args: tokens list of words, together forming a sentence. Returns: its embedding as a ndarray."""
        assert ndim == 1, "ndim has to be equal to 1 for skipthoughts embedding"
        sentence = " ".join(tokens)
        if sentence in self.cache_table:
            output_vector = self.cache_table[sentence]
        else:
            output_vector, = self.encode(self.st, [sentence, ], verbose=False)
            self.cache_table[sentence] = output_vector
        return output_vector
