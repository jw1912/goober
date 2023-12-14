pub mod activation;
mod matrix;
mod vector;

pub use matrix::Matrix;
pub use vector::{SparseVector, Vector};

pub trait OutputLayer {
    type Type;
    fn output_layer(&self) -> Self::Type;
}

pub trait FeedForwardNetwork: Sized {
    type Layers: OutputLayer;
    type InputType;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32);

    fn boxed_and_zeroed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    fn write_to_bin(&self, path: &str) {
        use std::io::Write;

        let mut file = std::fs::File::create(path).unwrap();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, std::mem::size_of_val(self));
            file.write_all(slice).unwrap();
        }
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers;

    fn out(&self, input: &Self::InputType) -> <Self::Layers as OutputLayer>::Type {
        self.out_with_layers(input).output_layer()
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: <Self::Layers as OutputLayer>::Type,
        layers: &Self::Layers,
    ) -> Self::InputType;
}
