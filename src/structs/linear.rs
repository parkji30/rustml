use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;

use crate::model::MLP;

pub fn prob_dens_func(x: f64, y: f64) -> f64{
    const std_x: f64 = 0.1;
    const std_y: f64 = 0.1;
    const A: f64 = 5.0;

    let x = x / 10.0;
    let y = y / 10.0;
    A * (-x.powi(2) / (2.0 / std_x.powi(2)) - y.powi(2) / (2 / std_y.powi(2))).exp()
}

pub struct LinearLayer{
    pub W: Array2<f64>;
    pub b: Array2<f64>;

    // Gradient of weights
    pub dW_dl: Option<Array2<f64>>;
    pub db_dl: Option<Array2<f64>>;

    // Gradient of output
    pub dy_dl: Option<Array2<f64>>;
}

impl MLP for LinearLayer {
    // get output of Linear Layer
    fn get_output(&self: x:)
}