#![deny(missing_docs)]

//! # Error Predictive Learning
//! Black-box learning algorithm using error prediction levels
//!
//! This is a very simple black-box learning algorithm which
//! uses higher order error prediction to improve
//! speed and accuracy of search to find local minima.
//!
//! See paper about [Error Predictive Learning](https://github.com/advancedresearch/path_semantics/blob/master/papers-wip/error-predictive-learning.pdf)
//!
//! ### Error prediction levels
//!
//! In error predictive learning, extra terms are added to the error
//! function such that the search algorithm must learn to predict error,
//! error in predicted error, and so on.
//! This information is used in a non-linear way to adapt search behavior,
//! which in turn affects error prediction etc.
//!
//! This algorithm is useful for numerical function approximation
//! of few variables due to high accuracy.
//!
//! ### Reset intervals
//!
//! In black-box learning, there are no assumptions about the function.
//! This makes it hard to use domain specific optimizations such as Newton's method.
//! The learning algorithm need to build up momentum in other ways.
//!
//! Counter-intuitively, forgetting the momentum from time to time
//! and rebuilding it might improve the search.
//! This is possible because re-learning momentum at a local point is relatively cheap.
//! The learning algorithm can takes advantage of local specific knowledge,
//! to gain the losses from forgetting the momentum.

pub mod utils;

/// Stores training settings.
#[derive(Copy, Clone, Debug)]
pub struct TrainingSettings {
    /// Acceptable accuracy in error.
    pub accuracy_error: f64,
    /// The minimum step value.
    ///
    /// When `error_predictions` is set to `0`, this is used as fixed step.
    pub step: f64,
    /// Maximum number of iterations.
    pub max_iterations: u64,
    /// The number of error prediction terms.
    ///
    /// More terms accelerate the search, but might lead to instability.
    pub error_predictions: usize,
    /// The interval to reset error predictions,
    /// in case they are far off or become unstable.
    pub reset_interval: u64,
    /// A factor greater than zero to prevent under or over-stepping.
    ///
    /// E.g. `0.95`
    ///
    /// This is used because predicted errors does not provide
    /// information about the gradient directly in the domain.
    /// Elasticity is used to estimate the gradient.
    pub elasticity: f64,
    /// Whether to print out result each reset interval.
    pub debug: bool,
}

/// Stores fit data.
#[derive(Clone, Debug)]
pub struct Fit {
    /// The error of the fit.
    pub error: f64,
    /// Weights of best fit (so far).
    pub weights: Vec<f64>,
    /// Error predictions weights.
    pub error_predictions: Vec<f64>,
    /// The number of iterations to produce the result.
    pub iterations: u64,
}

/// Trains to fit a vector of weights on a black-box function returning error.
///
/// Returns `Ok` if acceptable accuracy error was achieved.
/// Returns `Err` if exceeding max iterations or score was unchanged
/// for twice the reset interval.
pub fn train<F: Fn(&[f64]) -> f64>(
    settings: TrainingSettings,
    weights: &[f64],
    f: F
) -> Result<Fit, Fit> {
    let mut ws = vec![0.0; settings.error_predictions + weights.len()];
    for i in 0..weights.len() {
        ws[i + settings.error_predictions] = weights[i];
    }
    if settings.error_predictions > 0 {
        ws[0] = settings.step;
    }
    let eval = |ws: &[f64]| {
        let mut score = f(&ws[settings.error_predictions..]);
        for i in 0..settings.error_predictions {
            score += (score - ws[i]).abs();
        }
        score
    };
    let step = |ws: &[f64], i: usize| {
        settings.elasticity * if i + 1 < settings.error_predictions {
            // Use next error prediction level for change.
            ws[i + 1]
        } else if i + 1 == settings.error_predictions {
            // The last error prediction level uses normal step.
            settings.step
        } else if settings.error_predictions > 0 {
            // Adjust step to predicted error.
            ws[0]
        } else {
            settings.step
        }
    };
    let check = |w: &mut f64, i: usize| {
        if i < settings.error_predictions {
            if *w <= settings.step {*w = settings.step}
        }
    };
    let mut iterations = 0;
    // Keep track of last score to detect unchanged loop.
    let mut last_score: Option<f64> = None;
    let mut last_score_iterations = 0;
    loop {
        // Evaluate score without error predictions.
        let score = f(&ws[settings.error_predictions..]);
        if score <= settings.accuracy_error {
            return Ok(Fit {
                error: score,
                weights: ws[settings.error_predictions..].into(),
                error_predictions: ws[0..settings.error_predictions].into(),
                iterations,
            })
        } else if iterations >= settings.max_iterations ||
            last_score_iterations >= 2 * settings.reset_interval {
            return Err(Fit {
                error: score,
                weights: ws[settings.error_predictions..].into(),
                error_predictions: ws[0..settings.error_predictions].into(),
                iterations,
            })
        }
        if last_score == Some(score) {
            last_score_iterations += 1;
        } else {
            last_score_iterations = 0;
        }
        last_score = Some(score);
        // Reset error predictions.
        if iterations % settings.reset_interval == 0 {
            if settings.debug {
                println!("{:?}", Fit {
                    error: score,
                    weights: ws[settings.error_predictions..].into(),
                    error_predictions: ws[0..settings.error_predictions].into(),
                    iterations,
                });
            }
            for i in 0..settings.error_predictions {
                ws[i] = 0.0;
            }
            ws[0] = settings.step;
        }
        // Change eight weight in either direction and pick the best.
        // This also changes the error prediction weights.
        for i in 0..ws.len() {
            let score = eval(&ws);
            let old = ws[i];
            let step = step(&ws, i);
            ws[i] += step;
            check(&mut ws[i], i);
            let score_up = eval(&ws);
            ws[i] -= 2.0 * step;
            check(&mut ws[i], i);
            let score_down = eval(&ws);
            if score <= score_up && score <= score_down {
                ws[i] = old;
            } else if score_up < score_down {
                ws[i] += 2.0 * step;
            }
        }
        iterations += 1;
    }
}
