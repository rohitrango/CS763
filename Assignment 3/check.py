import torch
from Model import Model
from Linear import Linear
from ReLU import ReLU
from Criterion import Criterion

trial = Model()

layers = [0,0,0,0]
layers[0] = Linear(12,15)
layers[1] = ReLU()
layers[2] = Linear(15,6)
layers[3] = ReLU()

for layer in layers:
	trial.addLayer(layer)

input = torch.randn(8,12)
label = torch.tensor([0,5,1,2,2,4,3,5])
cross = Criterion()
trial.dispGradParam()
out = trial.forward(input)
print(out.size())
loss = cross.forward(input,label)
print(loss)
gradOut = cross.backward(out,label)
print(gradOut.size())
trial.backward(input,gradOut)
trial.dispGradParam()
trial.clearGradParam()
trial.dispGradParam()

