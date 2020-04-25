use error_predictive_learning::*;

pub fn lin(ws: &[f64]) -> f64 {
    utils::cmp(20, 0.0, 10.0,
        |x| x * (0.123456789 + x).sin() - 9.87654321,
        |x|  x * (ws[0] + x).sin() - ws[1])
}

fn main() {
    let settings = TrainingSettings {
        accuracy_error: 0.0,
        step: 0.000000000001,
        max_iterations: 1000000000000,
        error_predictions: 3,
        reset_interval: 40200 / 10,
        elasticity: 1.55,
        debug: true,
    };
    let weights = [0.12345683657383492, 9.876541862947374];
    println!("{:?}", train(settings, &weights, lin));
}
