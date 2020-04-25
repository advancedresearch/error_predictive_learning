use error_predictive_learning::*;

pub fn lin(ws: &[f64]) -> f64 {
    utils::cmp(20, 0.0, 10.0,
        |x| 0.12345678987654321 - 9.876543210123456789 * x,
        |x| ws[0] + ws[1] * x)
}

fn main() {
    let settings = TrainingSettings {
        accuracy_error: 0.0,
        step: 0.0000000000000000000000000001,
        max_iterations: 1000000000000,
        error_predictions: 3,
        reset_interval: 40200,
        elasticity: 1.55,
        debug: true,
    };
    let weights = [0.12345678970668006, -9.876543207050862];
    println!("{:?}", train(settings, &weights, lin));
}
