use goober::{activation::ReLU, layer::DenseConnected, FeedForwardNetwork, OutputLayer, Vector};

#[derive(FeedForwardNetwork)]
pub struct TestNet {
    l1: DenseConnected<ReLU, 768, 32>,
    l2: SubTestNet,
}

#[derive(FeedForwardNetwork)]
pub struct SubTestNet {
    l1: DenseConnected<ReLU, 32, 16>,
    l2: DenseConnected<ReLU, 16, 1>,
}

fn main() {
    let net = TestNet::boxed_and_zeroed();

    let input = Vector::from_raw([1.0; 768]);
    let output = net.out(&input);
    println!("{}", output[0])
}
