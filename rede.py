# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:23:01 2017

@author: alef1
"""

from numpy import array

def to_pattern(letter):
    return array([+1 if c=='X' else -1 for c in letter.replace('\n','')])

def display(pattern):
    from pylab import imshow, cm, show
    imshow(pattern.reshape((5,5)),cmap=cm.binary, interpolation='nearest')
    show()
    
def train(patterns):
    from numpy import zeros, outer, diag_indices 
    r,c = patterns.shape
    W = zeros((c,c))
    for p in patterns:
        W = W + outer(p,p)
    W[diag_indices(c)] = 0
    return W/r

def recall(W, patterns, steps=5):
    from numpy import vectorize, dot
    sgn = vectorize(lambda x: -1 if x<0 else +1)
    for _ in range(steps):        
        patterns = sgn(dot(patterns,W))
    return patterns
    
def hopfield_energy(W, patterns):
    from numpy import array, dot
    return array([-0.5*dot(dot(p.T,W),p) for p in patterns])


def main():
    A    = ".XXX.X...XXXXXXX...XX...X"
    Z    = "XXXXX...X...X...X...XXXXX"
    um   = ".XX....X....X....X....X.."
    zero = ".XXX.X...XX...XX...X.XXX."
    dois = ".XXX....X....X...X...XXX."
    tres = "XXXXX....X.XXXX....XXXXXX"
    quat = "X...XX...XXXXXX....X....X"
    cinc = "XXXXXX....XXXXX....XXXXXX"
    seis = "XXXXXX....XXXXXX...XXXXXX"
    sete = "XXXXX...X...X...X...X...."
    oito = "XXXXXX...XXXXXXX...XXXXXX"
    nove = "XXXXXX...XXXXXX....X....X"
    patterns = array([to_pattern(oito), to_pattern(nove), to_pattern(sete)])
    pesos = train(patterns)
    print(len(pesos[0]))
    result = recall(pesos,patterns[2],5)
    display(result)
    
    

if __name__ == "__main__":
    main()