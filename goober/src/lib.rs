pub mod activation;
pub mod layer;
mod matrix;
mod subnet;
mod vector;

pub use matrix::Matrix;
pub use subnet::SubNet;
pub use vector::Vector;

pub trait InputLayer {
    type Type;
    fn input_layer(&self) -> Self::Type;
}

pub trait OutputLayer {
    type Type;
    fn output_layer(&self) -> Self::Type;
}

pub trait NetworkUtils: OutputLayer + InputLayer {
    type Layers: OutputLayer + InputLayer;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32);

    fn zeroed() -> Self;
}

pub trait Network: NetworkUtils {
    fn out_with_layers(&self, input: <Self::Layers as InputLayer>::Type) -> Self::Layers;

    fn out(&self, input: <Self::Layers as InputLayer>::Type) -> <Self::Layers as OutputLayer>::Type {
        self.out_with_layers(input).output_layer()
    }
}

pub trait Trainable: Network {
    fn backprop(
        &self,
        feats: <Self::Layers as InputLayer>::Type,
        grad: &mut Self,
        out_err: <Self::Layers as OutputLayer>::Type,
        layers: Self::Layers,
    );
}
