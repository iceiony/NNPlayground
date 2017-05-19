ITERATIONS = 500
OUTPUT_FILE = 'cpu.csv'

import matplotlib.pyplot as plt
import dynet as dy
import numpy as np

def run_model():
    data_in = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    data_out = np.array([0, 1, 1, 0]).reshape(1, 4)


    m = dy.Model()
    sgd = dy.SimpleSGDTrainer(m, 1)

    pW = m.add_parameters((2, 2))
    pb = m.add_parameters(2)
    pV = m.add_parameters((1, 2))
    pa = m.add_parameters(1)

    errors = []
    for iter in range(ITERATIONS):
        dy.renew_cg()

        W = dy.parameter(pW)
        b = dy.parameter(pb)
        V = dy.parameter(pV)
        a = dy.parameter(pa)

        x = dy.inputTensor(data_in, batched=True)
        y = dy.inputTensor(data_out, batched=True)

        h = dy.tanh((W*x) + b)
        y_pred = dy.logistic((V*h) + a)
        loss = dy.binary_log_loss(y_pred, y)

        sum_loss = dy.sum_batches(loss) / 4 # pytorch devides by the number of records
        errors.append(sum_loss.scalar_value())
        sum_loss.backward()
        sgd.update()
    return errors

f = open(OUTPUT_FILE,'w')
print("Training Error: ")
for i in range(100):
    errors = run_model()
    print(', '.join(map(str,errors)), file = f)
    print(errors[-1])
f.close()

plt.plot(errors)
plt.savefig('error.png')
