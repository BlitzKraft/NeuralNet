#!/usr/bin/python
#To create a dataset

from pybrain.datasets import ClassificationDataSet as cd
from pybrain.utilities import percentError as pe
from pybrain.tools.shortcuts import buildNetwork as bn
from pybrain.supervised.trainers import BackpropTrainer as bt
from pybrain.structure.modules import SoftmaxLayer as sl

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

means = [(-1,0), (2,4), (3,1)]
cov = [diag([1,1]), diag([0.5, 1.2]), diag([1.5, 0.7])]

alldata = cd(2,1,nb_classes=3)

for n in xrange(400):
	for klass in range(3):
		input = multivariate_normal(means[klass], cov[klass])
		alldata.addSample(input, [klass])

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of patterns: ", len(trndata)
print "Input and output dimensions: ", trndata,indim, trndata.outdim
print "First sample(input, target, class): "
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = bn(trndata,indim, 5, trndata.outdim, outclass=sl)

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3., 6., 0.2)
X, Y = meshgrid(ticks, ticks)

griddata = cd(2,1, nb_classes=3)
for i in xrange(X.size):
	griddata.addSample([X.ravel()[i], Y.ravel()[i]],[0])
griddata._convertToOneOfMany() 

for i in range(20):
	trainer.trainEposhs(1)
	trnresult = pe(trainer.testOnClassData(), trndata['class'])
	tstresult = pe(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

	print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult

	out = fnn.activateOnDataSet(griddata)
	out = out.argmax(axis=1)
	out = out.reshape(X.shape)
	figure(1)
	ioff()
	clf()
	hold(True)
	for c in [0, 1, 2]: 
		here, _ = where(tstdata['class']==c)
		plot(tstdata['input'][here, 0], tstdata['input'][here,1],'o')
	if out.max()!=out.min():
		contourf(X,Y,out)
	ion()
	draw()

ioff()
show()