use goober::{Vector, activation::{Identity, ReLU}, layer::{SparseLayer, Layer}, NetworkUtils, Network, Trainable, SparseVector, OutputLayer};

#[repr(C)]
#[derive(Clone, Copy, NetworkUtils)]
pub struct SubNet {
    ft: SparseLayer<ReLU, 768, 16>,
}

impl Network for SubNet {
    type InputType = SparseVector<32>;

    fn out_with_layers(&self, input: Self::InputType) -> Self::Layers {
        Self::Layers {
            ft: self.ft.out(input)
        }
    }
}

impl Trainable for SubNet {
    fn backprop(
        &self,
        input: Self::InputType,
        grad: &mut Self,
        out_err: Vector<16>,
        layers: Self::Layers,
    ) {
        self.ft.backprop(&mut grad.ft, out_err, input, layers.ft);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PerspectiveNet {
    stm: SideNet,
    nstm: SideNet,
}

type SideLayers = <SideNet as NetworkUtils>::Layers;

impl PerspectiveNet {
    pub fn out_with_layers(
        &self,
        stm: SparseVector<32>,
        nstm: SparseVector<32>
    ) -> (SideLayers, SideLayers) {
        (self.stm.out_with_layers(stm), self.nstm.out_with_layers(nstm))
    }

    pub fn backprop(
        &self,
        stm: SparseVector<32>,
        nstm: SparseVector<32>,
        grad: &mut Self,
        out_err: f32,
        stm_layers: SideLayers,
        nstm_layers: SideLayers,
    ) {
        let err = Vector::from_raw([out_err; 1]);
        self.stm.backprop(stm, &mut grad.stm, err, stm_layers);
        self.nstm.backprop(nstm, &mut grad.nstm, err, nstm_layers);
    }
}

#[repr(C)]
#[derive(Clone, Copy, NetworkUtils)]
pub struct SideNet {
    ft: SparseLayer<ReLU, 768, 512>,
    l2: Layer<Identity, 512, 1>,
}

impl Network for SideNet {
    type InputType = SparseVector<32>;

    fn out_with_layers(&self, input: Self::InputType) -> Self::Layers {
        let ft = self.ft.out(input);
        Self::Layers {
            ft,
            l2: self.l2.out(ft),
        }
    }
}

impl Trainable for SideNet {
    fn backprop(
        &self,
        input: Self::InputType,
        grad: &mut Self,
        out_err: <Self::Layers as OutputLayer>::Type,
        layers: Self::Layers,
    ) {
        let next_err = self.l2.backprop(&mut grad.l2, out_err, layers.ft, layers.l2);
        self.ft.backprop(&mut grad.ft, next_err, input, layers.ft);
    }
}
