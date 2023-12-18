use std::marker::PhantomData;

use goober_core::{FeedForwardNetwork, OutputLayer, Vector, activation::Activation};

/// Applies a 1D Convolution from input dimension `M` to output dimension `N`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Conv1D<T, const M: usize, const N: usize> {
    weights: Vector<M>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T, const M: usize, const N: usize> Conv1D<T, M, N> {
    pub fn from_raw(weights: Vector<M>, bias: Vector<N>) -> Self {
        Self { weights, bias, phantom: PhantomData }
    }
}

pub struct Conv1DLayers<const N: usize> {
    out: Vector<N>,
}

impl<const N: usize> OutputLayer<Vector<N>> for Conv1DLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.out
    }
}

impl<T, const M: usize, const N: usize> FeedForwardNetwork for Conv1D<T, M, N>
where T: Activation
{
    type InputType = Vector<M>;
    type OutputType = Vector<N>;
    type Layers = Conv1DLayers<N>;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.weights.adam(g.weights, &mut m.weights, &mut v.weights, adj, lr);
    }

    fn backprop(
        &self,
        input: &Vector<M>,
        grad: &mut Self,
        out_err: Vector<N>,
        layers: &Conv1DLayers<N>,
    ) -> Vector<M> {
        unimplemented!()
    }

    fn out_with_layers(&self, input: &Vector<M>) -> Conv1DLayers<N> {
        let k = M - N + 1;

        let out = Vector::from_fn(|i| {
            let mut val = self.bias[i];
            for j in 0..k {
                val += input[i + j] * self.weights[j];
            }
            val
        });

        Conv1DLayers { out: out.activate::<T>() }
    }
}
