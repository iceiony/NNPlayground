#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;


float run_model(){
  // parameters
  const unsigned HIDDEN_SIZE = 2;
  const unsigned ITERATIONS = 500;


  ParameterCollection m;
  SimpleSGDTrainer trainer(m, 0.5);

  ComputationGraph cg;
  Parameter p_W, p_b, p_V, p_a;

  p_W = m.add_parameters({HIDDEN_SIZE, 2});
  p_b = m.add_parameters({HIDDEN_SIZE});
  p_V = m.add_parameters({1, HIDDEN_SIZE});
  p_a = m.add_parameters({1});

  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  // set x_values to change the inputs to the network
  Dim x_dim({2}, 4), y_dim({1}, 4);

  vector<dynet::real> y_values = {0.0, 1.0, 1.0, 0.0};
  vector<dynet::real> x_values = {1.0, 1.0, 
                                  1.0, 0.0,
                                  0.0, 1.0,
                                  0.0, 0.0};

  Expression x = input(cg, x_dim, &x_values);
  Expression y = input(cg, y_dim, &y_values);

  Expression h = logistic(W * x + b);
  Expression y_pred = logistic( V * h + a);

  y_pred = reshape(y_pred, Dim({4}, 1));
  y = reshape(y, Dim({4}, 1));

  Expression loss = binary_log_loss(y_pred, y);
  
  Expression sum_loss = sum_batches(loss);

  float my_loss = 1000;
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    my_loss = as_scalar(cg.forward(sum_loss)) / 4;
    cg.backward(sum_loss);
    trainer.update();
    //cerr << "E = " << my_loss << endl;
  }

  return my_loss;
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  for(unsigned i = 0; i < 1000; ++i){
    float loss = run_model();
    //cerr << loss << endl;
  }
}
