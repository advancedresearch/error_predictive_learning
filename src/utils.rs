//! Utility methods.

/// Compares two functions of a single variable.
pub fn cmp<F, G>(
    n: u32,
    start: f64,
    end: f64,
    f: F,
    g: G
) -> f64
    where F: Fn(f64) -> f64,
          G: Fn(f64) -> f64
{
    let mut sum = 0.0;
    for i in 0..n {
        let x = i as f64 / n as f64 * (end - start) + start;
        sum += (f(x) - g(x)).powi(2);
    }
    sum / n as f64
}
