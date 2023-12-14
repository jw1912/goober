use goober::{
    activation::ReLU,
    layer::{DenseConnected, SparseConnected},
    FeedForwardNetwork, OutputLayer, SparseVector,
};

#[derive(FeedForwardNetwork)]
pub struct TestNet {
    l1: SparseConnected<ReLU, 768, 32>,
    l2: SubTestNet,
}

#[derive(FeedForwardNetwork)]
pub struct SubTestNet {
    l1: DenseConnected<ReLU, 32, 16>,
    l2: DenseConnected<ReLU, 16, 1>,
}

#[test]
fn connected() {
    let net = TestNet::boxed_and_zeroed();

    let mut input = SparseVector::with_capacity(8);
    input.push(5);
    let _ = net.out(&input);
}
