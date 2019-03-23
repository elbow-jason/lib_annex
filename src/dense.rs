use super::activation::ActivatorFn;
use super::neuron::Neuron;
use super::num_type::Num;

pub type Matrix = Vec<Neuron>;

fn new_matrix(rows: usize, cols: usize, data: Vec<Num>, biases: Vec<Num>) -> Matrix {
    if rows == 0 {
        panic!("rows cannot be 0");
    }
    if cols == 0 {
        panic!("cols cannot be 0");
    }
    if data.len() == 0 {
        panic!("data cannot be empty");
    }
    if biases.len() != rows {
        panic!(
            "matrix bias size mismatch rows: {:?} length: {:?} biases: {:?}",
            rows,
            biases.len(),
            biases
        );
    }
    if data.len() != rows * cols {
        panic!(
            "matrix data size mismatch rows: {:?} cols: {:?} length: {:?} data: {:?}",
            rows,
            cols,
            data.len(),
            data
        );
    }
    data.chunks(cols)
        .zip(biases.iter())
        .map(|(weights, bias)| Neuron::build(weights.to_vec(), *bias))
        .collect()
}

fn matrix_rows(matrix: &Matrix) -> usize {
    matrix.len()
}

fn matrix_cols(matrix: &Matrix) -> usize {
    matrix[0].size()
}

pub struct Dense {
    matrix: Matrix,
    activation_derivative: Box<ActivatorFn>,
    learning_rate: f64,
    inputs: Vec<Num>,
}

impl Dense {
    pub fn build(
        rows: usize,
        cols: usize,
        data: Vec<Num>,
        biases: Vec<Num>,
        activation_derivative: Box<ActivatorFn>,
        learning_rate: f64,
    ) -> Dense {
        Dense {
            activation_derivative,
            learning_rate,
            matrix: new_matrix(rows, cols, data, biases),
            inputs: Vec::with_capacity(rows),
        }
    }

    pub fn rows(&self) -> usize {
        matrix_rows(&self.matrix)
    }

    pub fn cols(&self) -> usize {
        matrix_cols(&self.matrix)
    }

    pub fn weights(&self) -> Matrix {
        self.matrix.clone()
    }

    pub fn sums(&self) -> Vec<Num> {
        self.matrix.iter().map(|neuron| neuron.sum()).collect()
    }

    pub fn biases(&self) -> Vec<Num> {
        self.matrix.iter().map(|neuron| neuron.bias()).collect()
    }

    pub fn feedforward(&mut self, inputs: &[Num]) -> Vec<Num> {
        self.inputs = inputs.to_vec();
        for neuron in self.matrix.iter_mut() {
            neuron.feedforward(inputs);
        }
        self.sums()
    }

    pub fn backprop(&mut self, total_loss_pd: Num, loss_pds: &[Num]) -> Vec<Num> {
        let learning_rate = self.learning_rate;
        let activation_derivative = Box::new(&self.activation_derivative);
        let inputs = &self.inputs;
        self.matrix
            .iter_mut()
            .zip(loss_pds.iter())
            .flat_map(|(neuron, neuron_loss_pd)| {
                neuron.backprop(
                    inputs,
                    total_loss_pd,
                    *neuron_loss_pd,
                    learning_rate,
                    &activation_derivative,
                )
            })
            .collect()
    }
    // {next_loss_pd, neurons} =
    //   [get_neurons(layer), List.wrap(loss_pds)]
    //   |> Enum.zip()
    //   |> Enum.map(fn {neuron, loss_pd} ->
    //
    //   end)
    //   |> Enum.unzip()

    // {List.flatten(next_loss_pd), [], %Dense{layer | neurons: neurons}}
    // }

    // pub fn backprop(&mut self, total_loss_pd: Num, loss_pds: &[Num]) -> Vec<Num> {
    //     let mut index = 0;
    //     let mut sum_deriv;
    //     let mut delta_coeff;
    //     let rows = self.rows();

    //     let mut next_loss = Vec::with_capacity(rows);
    //     let input_array = Array::from_vec(self.inputs.clone());

    //     for mut row in self.matrix.genrows_mut() {
    //         sum_deriv = (self.activation_derivative)(self.sums[index]);
    //         delta_coeff = self.learning_rate * total_loss_pd * loss_pds[index];
    //         self.biases[index] += sum_deriv * delta_coeff * self.biases[index];
    //         Zip::from(&mut row)
    //             .and(input_array.view())
    //             .apply(|weight, input| {
    //                 next_loss.push(*weight * sum_deriv);
    //                 *weight -= input * sum_deriv * delta_coeff;
    //             });

    //         index += 1;
    //     }
    //     next_loss
    // }
}

pub fn total_loss_pd(outputs: Vec<Num>, labels: Vec<Num>) -> Num {
    let network_error = calc_network_error(outputs, labels);
    calculate_network_error_pd(network_error)
}

pub fn calculate_network_error_pd(network_error: Vec<Num>) -> Num {
    -2.0 * network_error.iter().sum::<f64>()
}

pub fn calc_network_error(net_outputs: Vec<Num>, labels: Vec<Num>) -> Vec<Num> {
    labels
        .iter()
        .zip(net_outputs.iter())
        .map(|(l, o)| l - o)
        .collect()
}

#[cfg(test)]
mod dense_tests {
    use super::super::activation::sigmoid_deriv;
    use super::super::num_type::Num;
    use super::{total_loss_pd, Dense};

    fn dense_fixture() -> Dense {
        let biases = vec![1.0, 1.0];
        let data = vec![1.0, 1.0, 1.0, 0.4, 0.4, 0.4];
        let activation_derivative = Box::new(sigmoid_deriv);
        let learning_rate = 0.05;
        Dense::build(2, 3, data, biases, activation_derivative, learning_rate)
    }

    #[test]
    fn feedforward_test() {
        let mut d: Dense = dense_fixture();
        let mut row_index: usize = 0;
        let expected_rows = vec![vec![1.0, 1.0, 1.0], vec![0.4, 0.4, 0.4]];
        for neuron in d.matrix.iter() {
            assert_eq!(neuron.clone_weights(), expected_rows[row_index]);
            row_index += 1;
        }
        let inputs = vec![0.3, 0.3, 0.3];
        let outputs = d.feedforward(&inputs);
        let expected_outputs = vec![1.9, 1.3599999999999999];
        assert_eq!(outputs, expected_outputs);
    }

    #[test]
    fn backprop_test() {
        let mut d: Dense = dense_fixture();
        let inputs = vec![0.1, 1.0, 0.0];
        let labels = vec![1.0, 0.0];
        let labels_count = labels.len();
        let outputs = d.feedforward(&inputs);
        let total_loss_pd_val = total_loss_pd(outputs, labels);
        let ones: Vec<Num> = (1..labels_count + 1).map(|_| 1.0).collect();
        assert_eq!(total_loss_pd_val, 5.08);
        let backprop_data = d.backprop(total_loss_pd_val, &ones);

        assert_eq!(d.inputs, inputs);

        let expected_sums = vec![2.1, 1.44];
        assert_eq!(d.sums(), expected_sums);

        let expected_biases = vec![0.9753125449806411, 0.9606666450864929];
        assert_eq!(d.biases(), expected_biases);

        let expected_row0 = vec![0.9975312544980641, 0.9753125449806411, 1.0];
        let expected_row1 = vec![0.3960666645086493, 0.36066664508649293, 0.4];
        let expected_rows = vec![expected_row0, expected_row1];
        let mut index = 0;
        for neuron in d.matrix.iter() {
            assert_eq!(neuron.clone_weights(), expected_rows[index]);
            index += 1;
        }

        let expected_backprop_data = vec![
            0.09719470480062539,
            0.09719470480062539,
            0.09719470480062539,
            0.06194229120237336,
            0.06194229120237336,
            0.06194229120237336,
        ];

        assert_eq!(backprop_data, expected_backprop_data);
    }
}
