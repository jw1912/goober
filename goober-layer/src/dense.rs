use std::marker::PhantomData;

use goober_core::{activation::Activation, FeedForwardNetwork, Matrix, OutputLayer, Vector};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DenseConnected<T: Activation, const M: usize, const N: usize> {
    weights: Matrix<N, M>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> std::ops::AddAssign<&DenseConnected<T, M, N>>
    for DenseConnected<T, M, N>
{
    fn add_assign(&mut self, rhs: &DenseConnected<T, M, N>) {
        self.weights += &rhs.weights;
        self.bias += rhs.bias;
    }
}

impl<T: Activation, const M: usize, const N: usize> DenseConnected<T, M, N> {
    pub const INPUT_SIZE: usize = M;
    pub const OUTPUT_SIZE: usize = N;

    pub const fn zeroed() -> Self {
        Self::from_raw(Matrix::zeroed(), Vector::zeroed())
    }

    pub const fn from_raw(weights: Matrix<N, M>, bias: Vector<N>) -> Self {
        Self {
            weights,
            bias,
            phantom: PhantomData,
        }
    }

    pub fn transpose_mul(&self, out: Vector<N>) -> Vector<M> {
        self.weights.transpose_mul(out)
    }
}

pub struct DenseConnectedLayers<const N: usize> {
    out: Vector<N>,
}

impl<const N: usize> OutputLayer for DenseConnectedLayers<N> {
    type Type = Vector<N>;
    fn output_layer(&self) -> Self::Type {
        self.out
    }
}

impl<T: Activation, const M: usize, const N: usize> FeedForwardNetwork for DenseConnected<T, M, N> {
    type Layers = DenseConnectedLayers<N>;
    type InputType = Vector<M>;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.weights
            .adam(&g.weights, &mut m.weights, &mut v.weights, adj, lr);

        self.bias.adam(g.bias, &mut m.bias, &mut v.bias, adj, lr);
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        Self::Layers {
            out: (self.weights * *input + self.bias).activate::<T>(),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        mut out_err: <Self::Layers as OutputLayer>::Type,
        layers: &Self::Layers,
    ) -> Self::InputType {
        out_err = out_err * layers.out.derivative::<T>();

        for (i, row) in grad.weights.iter_mut().enumerate() {
            *row += out_err[i] * *input;
        }

        grad.bias += out_err;
        self.transpose_mul(out_err)
    }
}
