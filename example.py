import numpy as np
import chainer
from chainer import ChainList, Variable, optimizers
import chainer.links as L
import chainer.functions as F


class MyChain(ChainList):
    def __init__(self):
        super(MyChain, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)


model = MyChain()
optimizer = chainer.optimizers.SGD()
optimizer.use_cleargrads(True)
optimizer.setup(model)

x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()
