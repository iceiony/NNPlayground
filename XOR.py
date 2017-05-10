import matplotlib.pyplot as plt
import numpy as np
from chainer.cuda import cupy as cp
import chainer
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

model = MLPXOR()

chainer.cuda.get_device_from_id(0).use()
model.to_gpu()
np = cp

data_in = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]]).astype('f')
data_out = np.array([0, 1, 1, 0]).astype('i').reshape((4, 1))


optimizer = optimizers.SGD(1.0)
optimizer.setup(model)

x = Variable(data_in)
y = Variable(data_out)

errors = []
for it in range(1, 5000):
    model.cleargrads()
    error = F.sigmoid_cross_entropy(model(x), y)
    error.backward()
    optimizer.update()
    # errors.append(error.data)


print('Train error ', error.data)
# model = F.sigmoid(model(x))
# print('Output :\n', np.round(cp.asnumpy(model.data)))
# print('Intended :\n', y.data)
# plt.plot(errors)
# plt.show()
#
# import chainer.computational_graph as c
# g = c.build_computational_graph(model)
# with open('./XOR.DOT', 'w') as o:
#    o.write(g.dump())
