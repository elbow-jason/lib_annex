extern crate ndarray;

pub mod activation;
pub mod dense;
pub mod neuron;
mod num_type;
// mod dense_test;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
