use goober_core::{FeedForwardNetwork, OutputLayer};

/// Adds two sub-networks that have common inputs and outputs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Add<A, B> {
    a: A,
    b: B,
}

impl<A, B> std::ops::AddAssign<&Add<A, B>> for Add<A, B>
where
    for<'a> A: FeedForwardNetwork + std::ops::AddAssign<&'a A>,
    for<'a> B: FeedForwardNetwork + std::ops::AddAssign<&'a B>,
{
    fn add_assign(&mut self, rhs: &Add<A, B>) {
        self.a += &rhs.a;
        self.b += &rhs.b;
    }
}

pub struct AddLayers<A, B>
where
    A: FeedForwardNetwork,
    B: FeedForwardNetwork,
{
    a: A::Layers,
    b: B::Layers,
}

impl<A, B> OutputLayer<A::OutputType> for AddLayers<A, B>
where
    A: FeedForwardNetwork,
    B: FeedForwardNetwork<OutputType = A::OutputType>,
    A::OutputType: std::ops::Add<A::OutputType, Output = A::OutputType>,
{
    fn output_layer(&self) -> A::OutputType {
        self.a.output_layer() + self.b.output_layer()
    }
}

impl<A, B> FeedForwardNetwork for Add<A, B>
where
    A: FeedForwardNetwork,
    B: FeedForwardNetwork<InputType = A::InputType, OutputType = A::OutputType>,
    A::OutputType: std::ops::Add<A::OutputType, Output = A::OutputType>,
    A::InputType: std::ops::Add<A::InputType, Output = A::InputType>,
{
    type InputType = A::InputType;
    type OutputType = A::OutputType;
    type Layers = AddLayers<A, B>;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.a.adam(&g.a, &mut m.a, &mut v.a, adj, lr);
        self.b.adam(&g.b, &mut m.b, &mut v.b, adj, lr);
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        Self::Layers {
            a: self.a.out_with_layers(input),
            b: self.b.out_with_layers(input),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        let a_back = self
            .a
            .backprop(input, &mut grad.a, out_err.clone(), &layers.a);
        let b_back = self.b.backprop(input, &mut grad.b, out_err, &layers.b);
        a_back + b_back
    }
}

impl<A, B> Add<A, B> {
    pub const fn from_raw(a: A, b: B) -> Self {
        Self { a, b }
    }
}
