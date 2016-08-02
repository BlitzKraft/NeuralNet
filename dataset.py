#!/usr/bin/python
#To create a dataset

from pybrain.datasets import ClassificationDataSet as cd
from pybrain.utilities import percentError as pe
from pybrain.tools.shortcuts import buildNetwork as bn
from pybrain.supervised.trainers import BackpropTrainer as bt
from pybrain.structure.modules import SoftmaxLayer as sl
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import matplotlib
import matplotlib.pylab as mp

#Means around which the random samples are generated
means = [(-1,0), (2,4), (3,1)]

#Covariance matrix for the random data, one per mean
cov = [diag([1,1]), diag([0.5, 1.2]), diag([1.5, 0.7])]

#Variable that holds all the data generated
alldata = cd(2,1,nb_classes=3)

for n in xrange(400):
	for klass in range(3):
		input = multivariate_normal(means[klass], cov[klass])
		alldata.addSample(input, [klass])

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample(input, target, class): "
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = bn(trndata.indim, 5, trndata.outdim, outclass=sl)

trainer = bt( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#arange start stop step
ticks = arange(-3., 6., 0.2)

#returns coordinate matrices from coordinate vector
X, Y = meshgrid(ticks, ticks)

griddata = cd(2,1, nb_classes=3)
for i in xrange(X.size):
    #ravel creates a vector from a matrix
    #by unwrapping it.
    griddata.addSample([X.ravel()[i], Y.ravel()[i]],[0])
griddata._convertToOneOfMany() 

for i in range(20):
    #fnn bt
    trainer.trainEpochs(1)
    trnresult = pe(trainer.testOnClassData(), trndata['class'])
    tstresult = pe(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

    print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult

    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)
    out = out.reshape(X.shape)
    mp.figure(1)
    mp.ioff()
    mp.clf()
    mp.hold(True)
    for c in [0, 1, 2]: 
        here, _ = where(tstdata['class']==c)
        mp.plot(tstdata['input'][here, 0], tstdata['input'][here,1],'o')
    if out.max()!=out.min():
            mp.contourf(X,Y,out)
    mp.ion()
    mp.draw()

mp.ioff()
mp.show()
