use super::activation::Activator;
use super::num_type::Num;

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Num>,
    _bias: Num,
    _sum: Num,
    output: Num,
}

impl Neuron {
    pub fn build(weights: Vec<Num>, bias: Num) -> Neuron {
        Neuron {
            weights,
            _bias: bias,
            _sum: 0.0,
            output: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.weights.len()
    }

    pub fn bias(&self) -> Num {
        self._bias
    }

    pub fn clone_weights(&self) -> Vec<Num> {
        self.weights.clone()
    }

    pub fn feedforward(&mut self, inputs: &[Num]) {
        let summed_weights: Num = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum();

        self._sum = self._bias + summed_weights;
    }

    pub fn sum(&self) -> Num {
        self._sum
    }

    pub fn backprop(
        &mut self,
        inputs: &[Num],
        total_loss_pd: Num,
        neuron_loss_pd: Num,
        learning_rate: Num,
        activator: &Activator,
    ) -> Vec<Num> {
        let delta_coeff = learning_rate * total_loss_pd * neuron_loss_pd;
        let sum_deriv = activator.derivative(self._sum);
        println!("total_loss_pd {:?}", total_loss_pd);
        println!("neuron_loss_pd {:?}", neuron_loss_pd);
        println!("learning_rate {:?}", learning_rate);
        println!("activator {:?}", activator);
        println!("sum_deriv {:?}", sum_deriv);
        println!("_sum {:?}", self._sum);
        println!("delta_coeff {:?}", delta_coeff);
        self._bias -= calc_delta(self._bias, sum_deriv, delta_coeff);
        inputs
            .iter()
            .zip(self.weights.iter_mut())
            .map(|(input, weight)| {
                let next_loss_pd = *weight * sum_deriv;
                *weight -= calc_delta(*input, sum_deriv, delta_coeff);
                next_loss_pd
            })
            .collect()
    }
}

fn calc_delta(input: Num, sum_deriv: Num, delta_coeff: Num) -> Num {
    input * sum_deriv * delta_coeff
}

#[cfg(test)]
mod neuron_test {
    use super::super::activation::Activator;
    use super::Neuron;

    #[test]
    fn build_test() {
        let weights = vec![1.0, 1.0, 2.0];
        let bias = 1.1;

        let n = Neuron::build(weights.clone(), bias);
        assert_eq!(n.clone_weights(), weights.clone());
    }

    #[test]
    fn feedforward_test() {
        let weights = vec![1.0, 0.0, -1.1];
        let bias = 1.0;
        let mut n = Neuron::build(weights.clone(), bias);
        assert_eq!(n.size(), 3);

        let inputs = vec![1.0, 0.9, 0.0];
        n.feedforward(&inputs);
        assert_eq!(n.clone_weights(), weights.clone());
        assert_eq!(n.sum(), 2.0);
    }

    #[test]
    fn backprop_test() {
        let weights = vec![1.0, 0.0, -1.1];
        let bias = 1.0;
        let mut n = Neuron::build(weights.clone(), bias);
        let inputs = vec![1.0, 0.9, 0.0];
        n.feedforward(&inputs);
        assert_eq!(n.sum(), 2.0);
        let total_loss_pd = 0.5;
        let neuron_loss_pd = 0.3;
        let learning_rate = 0.05;
        let activator = &Activator::Sigmoid;
        let next_loss_pds = n.backprop(
            &inputs,
            total_loss_pd,
            neuron_loss_pd,
            learning_rate,
            activator,
        );
        let expected_weights = vec![0.9992125481094737, -7.087067014736697e-4, -1.1];
        assert_eq!(n.clone_weights(), expected_weights);

        let expected_bias = 0.9992125481094737;
        assert_eq!(n.bias(), expected_bias);

        let expected_next_loss_pds = vec![0.10499358540350662, 0.0, -0.11549294394385728];
        assert_eq!(next_loss_pds, expected_next_loss_pds);
    }

}
