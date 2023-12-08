pub mod activation;
pub mod layer;
mod matrix;
mod subnet;
mod vector;

pub use matrix::Matrix;
pub use subnet::SubNet;
pub use vector::Vector;

pub trait FinalLayer<T> {
    fn output_layer(&self) -> T;
}

pub trait NetworkUtils: std::marker::Sized + std::ops::AddAssign<Self> {
    type OutputType;
    type Layers: FinalLayer<Self::OutputType>;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32);

    fn zeroed() -> Self;
}

pub trait Network<InputType>: NetworkUtils {
    fn out_with_layers(&self, input: InputType) -> Self::Layers;

    fn out(&self, input: InputType) -> Self::OutputType {
        self.out_with_layers(input).output_layer()
    }
}

pub trait Trainable<InputType>: Network<InputType> {
    fn backprop(&self, feats: InputType, grad: &mut Self, out_err: Self::OutputType, layers: Self::Layers);
}
