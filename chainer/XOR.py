CUDA = True
OUTPUT_FILE = 'xpu.csv'
ITERATIONS = 500


import matplotlib.pyplot as plt
import chainer
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import chainer.initializers as initializers

if CUDA:
    print('GPU run')
    from chainer.cuda import cupy as xp
else:
    print('CPU run')
    import numpy as xp

class MLPXOR(Chain):
    def __init__(self):
        super(MLPXOR, self).__init__(
            l1=L.Linear(2, 2, initialW=initializers.Normal(scale=1)),
            l2=L.Linear(2, 1, initialW=initializers.Normal(scale=1)),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        return self.l2(h1) #linear output for sigmoid_cross_entropy()

def run_model():
    model = MLPXOR()

    if CUDA:
        chainer.cuda.get_device_from_id(0).use()
        model.to_gpu()

    data_in = xp.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]]).astype('f')
    data_out = xp.array([0, 1, 1, 0]).astype('i').reshape((4, 1))

    optimizer = optimizers.SGD(1.0)
    optimizer.setup(model)

    x = Variable(data_in)
    y = Variable(data_out)

    errors = []
    for it in range(1, ITERATIONS):
        model.cleargrads()
        loss = F.sigmoid_cross_entropy(model(x), y)
        errors.append(loss.data)
        loss.backward()
        optimizer.update()
    return errors


f = open(OUTPUT_FILE, 'w')
print('Train errors: ')
for i in range(1):
    errors = run_model()
    print(', '.join(map(str,errors)), file=f)
    print(errors[-1])
f.close()

if CUDA:
    errors = [xp.asnumpy(x) for x in errors]
plt.plot(errors)
plt.savefig('error.png')

