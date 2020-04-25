use error_predictive_learning::*;

pub fn lin(ws: &[f64]) -> f64 {
    utils::cmp(20, 0.0, 6.283185307179586,
        |x| x.sin(),
        |x| (x - 0.25 * ws[0]).cos())
}

fn main() {
    let settings = TrainingSettings {
        accuracy_error: 0.0,
        step: 0.00000000000000000000001,
        max_iterations: 1000000000000,
        error_predictions: 3,
        reset_interval: 40200,
        elasticity: 1.55,
        debug: true,
    };
    let weights = [6.283185176129033];
    println!("{:?}", train(settings, &weights, lin));
}
