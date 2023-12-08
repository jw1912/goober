pub mod activation;
pub mod layer;
mod matrix;
mod vector;

pub use matrix::Matrix;
pub use vector::{Vector, SparseVector};

pub use goober_derive::NetworkUtils;

pub trait InputLayer {
    type Type;
}

pub trait OutputLayer {
    type Type;
    fn output_layer(&self) -> Self::Type;
}

pub trait NetworkUtils {
    type Layers: OutputLayer;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32);

    fn zeroed() -> Self;
}

pub trait Network: NetworkUtils {
    type InputType;

    fn out_with_layers(&self, input: Self::InputType) -> Self::Layers;

    fn out(&self, input: Self::InputType) -> <Self::Layers as OutputLayer>::Type {
        self.out_with_layers(input).output_layer()
    }
}

pub trait Trainable: Network {
    fn backprop(
        &self,
        input: Self::InputType,
        grad: &mut Self,
        out_err: <Self::Layers as OutputLayer>::Type,
        layers: Self::Layers,
    );
}
