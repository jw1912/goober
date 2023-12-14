use goober::{
    activation::ReLU,
    layer::{Add, DenseConnected, SparseConnected},
    FeedForwardNetwork, SparseVector,
};

#[derive(FeedForwardNetwork)]
pub struct TestNet {
    l1: Add<SparseConnected<ReLU, 768, 1>, SubTestNet>,
}

#[derive(FeedForwardNetwork)]
pub struct SubTestNet {
    l1: SparseConnected<ReLU, 768, 16>,
    l2: DenseConnected<ReLU, 16, 1>,
}

#[test]
fn add() {
    let net = TestNet::boxed_and_zeroed();

    let mut input = SparseVector::with_capacity(8);
    input.push(5);
    let _ = net.out(&input);
}