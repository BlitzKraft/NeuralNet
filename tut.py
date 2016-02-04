#!/usr/bin/python

from pybrain.structure import FeedForwardNetwork as ffn
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection as fc
from pybrain.structure import RecurrentNetwork as rn

n = rn()

n.addInputModule(LinearLayer(2, name = 'in'))
n.addModule(SigmoidLayer(3, name = 'hidden'))
n.addOutputModule(LinearLayer(1, name = 'out'))

n.addConnection(fc(n['in'], n['hidden'], name='c1'))
n.addConnection(fc(n['hidden'], n['out'], name='c2'))

n.addRecurrentConnection(fc(n['hidden'],n['hidden'], name='c3'))

n.sortModules()

print n.params
