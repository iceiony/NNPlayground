import matplotlib.pyplot as plt
import numpy as np
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L

class MLPXOR(Chain):
    def __init__(self):
        super(MLPXOR, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, 2),
            l2=L.Linear(None, 1),
        )

    def __call__(self, x):
        h1 = F.leaky_relu(self.l1(x))
        return self.l2(h1)


data_in = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]]).astype('f')
data_out = np.array([0, 1, 1, 0]).astype('i').reshape((4, 1))

model = MLPXOR()

optimizer = optimizers.SGD(0.5)
optimizer.setup(model)

x = Variable(data_in)
y = Variable(data_out)

errors = []
for it in range(1, 500):
    model.cleargrads()
    error = F.sigmoid_cross_entropy(model(x), y)
    error.backward()
    optimizer.update()
    errors.append(error.data)


model = F.sigmoid(model(x))
print('Train error ', errors[-1])
print('Output :\n', np.round(model.data))
print('Intended :\n', y.data)
plt.plot(errors)
plt.show()

import chainer.computational_graph as c
g = c.build_computational_graph(model)
with open('./XOR.DOT', 'w') as o:
   o.write(g.dump())
