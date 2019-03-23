use super::num_type::Num;
pub type Activator = Fn(Num) -> Num;

pub fn relu(n: Num) -> Num {
    relu_with_threshold(n, 0.0)
}

pub fn relu_deriv(n: Num) -> Num {
    relu_deriv_with_threshold(n, 0.0)
}

pub fn relu_deriv_with_threshold(n: Num, threshold: Num) -> Num {
    if n > threshold {
        1.0
    } else {
        0.0
    }
}

fn relu_with_threshold(n: Num, threshold: Num) -> Num {
    if n > threshold {
        n
    } else {
        threshold
    }
}

pub fn sigmoid(n: Num) -> Num {
    1.0 / (1.0 + ((-n).exp()))
}

pub fn sigmoid_deriv(x: Num) -> Num {
    let fx = sigmoid(x);
    fx * (1.0 - fx)
}

pub fn tanh(n: Num) -> Num {
    n.tanh()
}

pub fn tanh_deriv(x: Num) -> Num {
    1.0 - x.tanh().powi(2)
}

#[cfg(test)]
mod test {
    use super::{relu, relu_deriv, relu_with_threshold, sigmoid, sigmoid_deriv, tanh, tanh_deriv};

    #[test]
    fn relu_test() {
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(0.5), 0.5);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(-10.0), 0.0);
    }

    #[test]
    fn relu_deriv_test() {
        assert_eq!(relu_deriv(10.0), 1.0);
        assert_eq!(relu_deriv(1.0), 1.0);
        assert_eq!(relu_deriv(0.1), 1.0);
        assert_eq!(relu_deriv(0.0), 0.0);
        assert_eq!(relu_deriv(-1.0), 0.0);
        assert_eq!(relu_deriv(-0.0001), 0.0);
    }

    #[test]
    fn relu_with_threshold_test() {
        assert_eq!(relu_with_threshold(1.0, 2.0), 2.0);
        assert_eq!(relu_with_threshold(0.5, 2.0), 2.0);
        assert_eq!(relu_with_threshold(0.0, 2.0), 2.0);
        assert_eq!(relu_with_threshold(-10.0, 2.0), 2.0);
        assert_eq!(relu_with_threshold(10.0, 2.0), 10.0);
        assert_eq!(relu_with_threshold(1.9, 2.0), 2.0);
        assert_eq!(relu_with_threshold(2.1, 2.0), 2.1);
    }

    #[test]
    fn sigmoid_test() {
        assert_eq!(sigmoid(1.0), 0.7310585786300049);
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(-1.0), 0.2689414213699951);
    }

    #[test]
    fn sigmoid_deriv_test() {
        assert_eq!(sigmoid_deriv(1.0), 0.19661193324148185);
        assert_eq!(sigmoid_deriv(0.0), 0.25);
        assert_eq!(sigmoid_deriv(-1.0), 0.19661193324148185);
    }

    #[test]
    fn tanh_test() {
        assert_eq!(tanh(1.0), 0.7615941559557649);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(-1.0), -0.7615941559557649);
    }

    #[test]
    fn tanh_deriv_test() {
        assert_eq!(tanh_deriv(1.0), 0.41997434161402614);
        assert_eq!(tanh_deriv(0.0), 1.0);
        assert_eq!(tanh_deriv(-1.0), 0.41997434161402614);
    }

}
