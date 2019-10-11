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
    sgd = dy.SimpleSGDTrainer(m, 0.5)

    W = m.add_parameters((2, 2))
    b = m.add_parameters(2)

    V = m.add_parameters((1, 2))
    a = m.add_parameters(1)

    errors = []
    for iter in range(ITERATIONS):
        dy.renew_cg()

        x = dy.inputTensor(data_in, batched=True)
        y = dy.inputTensor(data_out)

        h = dy.logistic(W * x + b)
        y_pred = dy.logistic((V*h) + a)
        y_pred = dy.reshape(y_pred, y.dim()[0])
        loss = dy.binary_log_loss(y_pred, y)

        errors.append(loss.scalar_value() / 4)
        loss.backward()
        sgd.update()
    return errors

f = open(OUTPUT_FILE,'w')
print("Training Error: ")
for i in range(100):
    errors = run_model()
    print(', '.join(map(str,errors)), file = f)
    #print(errors[-1])
    if i % 10 == 1:
        print( i , '%')

f.close()

plt.plot(errors)
plt.savefig('error.png')
