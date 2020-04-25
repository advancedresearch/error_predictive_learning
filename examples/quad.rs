use error_predictive_learning::*;

pub fn lin(ws: &[f64]) -> f64 {
    utils::cmp(20, 0.0, 10.0,
        |x| 0.123456789 - 9.87654321 * x + 0.123456789 * x * x,
        |x| ws[0] + ws[1] * x + ws[2] * x * x)
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
    let weights = [0.12345666679385302, -9.876543156097174, 0.12345678422789191];
    println!("{:?}", train(settings, &weights, lin));
}
