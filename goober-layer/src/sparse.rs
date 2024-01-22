use std::marker::PhantomData;

use goober_core::{
    activation::Activation, FeedForwardNetwork, Matrix, OutputLayer, SparseVector, Vector,
};

/// Fully-Connected layer with sparse input.
/// - `T` is the activation function used.
/// - `M` is the size of the input vector.
/// - `N` is the size of the output vector.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SparseConnected<T: Activation, const M: usize, const N: usize> {
    weights: Matrix<M, N>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> std::ops::AddAssign<&SparseConnected<T, M, N>>
    for SparseConnected<T, M, N>
{
    fn add_assign(&mut self, rhs: &SparseConnected<T, M, N>) {
        self.weights += &rhs.weights;
        self.bias += rhs.bias;
    }
}

impl<T: Activation, const M: usize, const N: usize> SparseConnected<T, M, N> {
    pub fn weights_row(&self, idx: usize) -> Vector<N> {
        self.weights[idx]
    }

    pub fn weights_row_mut(&mut self, idx: usize) -> &mut Vector<N> {
        &mut self.weights[idx]
    }

    pub fn bias(&self) -> Vector<N> {
        self.bias
    }

    pub fn bias_mut(&mut self) -> &mut Vector<N> {
        &mut self.bias
    }

    pub const fn zeroed() -> Self {
        Self::from_raw(Matrix::zeroed(), Vector::zeroed())
    }

    pub const fn from_raw(weights: Matrix<M, N>, bias: Vector<N>) -> Self {
        Self {
            weights,
            bias,
            phantom: PhantomData,
        }
    }

    pub fn from_fn<W: FnMut(usize, usize) -> f32, B: FnMut(usize) -> f32>(w: W, b: B) -> Self {
        Self {
            weights: Matrix::from_fn(w),
            bias: Vector::from_fn(b),
            phantom: PhantomData,
        }
    }
}

pub struct SparseConnectedLayers<const N: usize> {
    out: Vector<N>,
}

impl<const N: usize> OutputLayer<Vector<N>> for SparseConnectedLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.out
    }
}

impl<T: Activation, const M: usize, const N: usize> FeedForwardNetwork
    for SparseConnected<T, M, N>
{
    type InputType = SparseVector;
    type OutputType = Vector<N>;
    type Layers = SparseConnectedLayers<N>;

    fn adam(&mut self, grad: &Self, momentum: &mut Self, velocity: &mut Self, adj: f32, lr: f32) {
        self.weights.adam(
            &grad.weights,
            &mut momentum.weights,
            &mut velocity.weights,
            adj,
            lr,
        );

        self.bias
            .adam(grad.bias, &mut momentum.bias, &mut velocity.bias, adj, lr);
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        let mut res = self.bias;

        for &feat in input.iter() {
            res += self.weights[feat];
        }

        Self::Layers {
            out: res.activate::<T>(),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        mut out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        out_err = out_err * layers.out.derivative::<T>();

        for &feat in input.iter() {
            grad.weights[feat] += out_err;
        }

        grad.bias += out_err;
        SparseVector::with_capacity(0)
    }
}

#[cfg(test)]
mod test {
    use super::SparseConnected;

    #[test]
    fn sparse_connected() {
        use goober_core::{activation::ReLU, FeedForwardNetwork, Matrix, SparseVector, Vector};

        let layer: SparseConnected<ReLU, 3, 3> = SparseConnected::from_raw(
            Matrix::from_raw([
                Vector::from_raw([1.0, 1.0, 0.0]),
                Vector::from_raw([1.0, 1.0, 1.0]),
                Vector::from_raw([1.0, 0.0, 1.0]),
            ]),
            Vector::from_raw([0.1, 0.1, 0.2]),
        );

        let mut input = SparseVector::with_capacity(8);
        input.push(0);
        input.push(1);
        input.push(2);

        let expected = Vector::from_raw([3.1, 2.1, 2.2]);
        assert_eq!(expected, layer.out(&input));
    }
}
