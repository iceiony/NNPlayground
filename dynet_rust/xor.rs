extern crate dynet;

use dynet::*;

fn run_model() -> f32 {
    let hidden_size = 2;
    let iterations = 500;

    let mut m = ParameterCollection::new();
    let mut trainer = SimpleSGDTrainer::new(&mut m, 0.5);

    let mut cg = ComputationGraph::new();
    let initializer = ParameterInitGlorot::default();

    let mut p_w = m.add_parameters([hidden_size, 2], &initializer);
    let mut p_b = m.add_parameters([hidden_size], &initializer);
    let mut p_v = m.add_parameters([1, hidden_size], &initializer);
    let mut p_a = m.add_parameters([1], &initializer);

    let y_values = [0.0, 1.0, 1.0, 0.0]; 
    let x_values = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let mut total_loss = 1000.0;
    for _iter in 0..iterations {
        cg.clear();

        let w = parameter(&mut cg, &mut p_w);
        let b = parameter(&mut cg, &mut p_b);
        let v = parameter(&mut cg, &mut p_v);
        let a = parameter(&mut cg, &mut p_a);

        let x = input(&mut cg, ([2], 4), &x_values);
        let mut y = input(&mut cg, ([1], 4), &y_values);

        let h = logistic(&w * &x + &b);
        let mut y_pred = logistic(&v * &h + &a);

        y_pred = reshape(y_pred, ([4], 1));
        y = reshape(y, ([4], 1));

        let loss = binary_log_loss(y_pred, y);

        total_loss = cg.forward(&loss).as_scalar();
        cg.backward(&loss);
        trainer.update();

        //cg.print_graphviz();
    }

    total_loss
}

fn main() {
    dynet::initialize(&mut DynetParams::from_args(true));

    for _i in 0..1000 {
        let _loss = run_model();
        //println!("E = {}", loss);
    }
}

