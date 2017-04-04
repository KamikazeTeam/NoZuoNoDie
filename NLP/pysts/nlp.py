"""
NLP preprocessing tools for sentences.

Currently, this just tags the token sequences by some trivial boolean flags
that denote some token characteristics and sentence-sentence overlaps.

In principle, this module could however include a lot more sophisticated
NLP tagging pipelines, or loading precomputed such data.
"""

import numpy as np
import re

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

flagsdim = 4

def Sentence_flags(s0, s1, s0pad, s1pad):
    """ For sentence lists s0, s1, generate numpy tensor
    (#sents, spad, flagsdim) that contains a sparse indicator vector of
    various token properties.  It is meant to be concatenated to the token
    embedding. """

    def gen_iflags(s, spad):
        iflags = []
        for i in range(len(s)):
            iiflags = [[False, False] for j in range(spad)]
            for j, t in enumerate(s[i]):
                if j >= spad:
                    break
                number = False
                capital = False
                if re.match('^[0-9\W]*[0-9]+[0-9\W]*$', t):
                    number = True
                if j > 0 and re.match('^[A-Z]', t):
                    capital = True
                iiflags[j] = [number, capital]
            iflags.append(iiflags)
        return iflags

    def gen_mflags(s0, s1, s0pad):
        """ generate flags for s0 that represent overlaps with s1 """
        mflags = []
        for i in range(len(s0)):
            mmflags = [[False, False] for j in range(s0pad)]
            for j in range(min(s0pad, len(s0[i]))):
                unigram = False
                bigram = False
                for k in range(len(s1[i])):
                    if s0[i][j].lower() != s1[i][k].lower():
                        continue
                    # do not generate trivial overlap flags, but accept them as part of bigrams                    
                    if s0[i][j].lower() not in stop and not re.match('^\W+$', s0[i][j]):
                        unigram = True
                    try:
                        if s0[i][j+1].lower() == s1[i][k+1].lower():
                            bigram = True
                    except IndexError:
                        pass
                mmflags[j] = [unigram, bigram]
            mflags.append(mmflags)
        return mflags

    # individual flags (for understanding)
    iflags0 = gen_iflags(s0, s0pad)
    iflags1 = gen_iflags(s1, s1pad)

    # s1-s0 match flags (for attention)
    mflags0 = gen_mflags(s0, s1, s0pad)
    mflags1 = gen_mflags(s1, s0, s1pad)

    def gen_pos(s, spad):
        pflags = []
        for si in s:
            pos=nltk.pos_tag(si)
            def toint(string):
                if string=='CC': return 1
                if string=='CD': return 2
                if string=='DT': return 3
                if string=='EX': return 4
                if string=='FW': return 5
                if string=='IN': return 6
                if string=='JJ': return 7
                if string=='JJR': return 8
                if string=='JJS': return 9
                if string=='LS': return 10
                if string=='MD': return 11
                if string=='NN': return 12
                if string=='NNS': return 13
                if string=='NNP': return 14
                if string=='NNPS': return 15
                if string=='PDT': return 16
                if string=='POS': return 17
                if string=='PRP': return 18
                if string=='PRP$': return 19
                if string=='RB': return 20
                if string=='RBR': return 21
                if string=='RBS': return 22
                if string=='RP': return 23
                if string=='SYM': return 24
                if string=='TO': return 25
                if string=='UH': return 26
                if string=='VB': return 27
                if string=='VBD': return 28
                if string=='VBG': return 29
                if string=='VBN': return 30
                if string=='VBP': return 31
                if string=='VBZ': return 32
                if string=='WDT': return 33
                if string=='WP': return 34
                if string=='WP$': return 35
                if string=='WRB': return 36
                return 0
            posvec = [[0] for i in range(spad)]
            for i in range(len(pos)):
                if i >= spad: break
                posvec[i] = [float(toint(pos[i][1]))/36.0]
            pflags.append(posvec)
        return pflags

    pflags0 = gen_pos(s0, s0pad)
    pflags1 = gen_pos(s1, s1pad)

    return [np.dstack((iflags0, mflags0)), np.dstack((iflags1, mflags1))]

    return [np.dstack((iflags0, mflags0, pflags0)),
            np.dstack((iflags1, mflags1, pflags1))]
