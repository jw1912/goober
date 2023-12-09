use goober::{
    activation::{Identity, ReLU},
    layer::{Layer, SparseLayer},
    FeedForwardNetwork,
};

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: SparseLayer<ReLU, 768, 16>,
}

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SideNet {
    ft: SparseLayer<ReLU, 768, 512>,
    l2: Layer<Identity, 512, 1>,
}
