use proc_macro2::{TokenStream, Ident, Span};
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(FeedForwardNetwork)]
pub fn network_utils(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let layer_name = Ident::new(
        (name.to_string() + "Layer").as_str(),
        Span::call_site(),
    );

    let add_impl = gen_add_impl(&input.data);
    let layer_fields = gen_layer_fields(&input.data);
    let output_type = gen_output_type(&input.data);
    let input_type = gen_input_type(&input.data);
    let adam_expr = gen_adam_expr(&input.data);
    let zeroed_expr = gen_zeroed_expr(&input.data);
    let layer_exprs = gen_layer_exprs(&input.data);
    let layer_exprs_fields = gen_layer_exprs_fields(&input.data);
    let backprop_exprs = gen_backprop_exprs(&input.data);

    let expanded = quote! {
        impl std::ops::AddAssign<& #name> for #name {
            fn add_assign(&mut self, rhs: & #name) {
                #add_impl
            }
        }

        pub struct #layer_name {
            #layer_fields
        }

        impl goober::OutputLayer for #layer_name {
            #output_type
        }

        impl goober::FeedForwardNetwork for #name {
            type Layers = #layer_name;
            type InputType = #input_type;

            fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
                #adam_expr
            }

            fn zeroed() -> Self {
                Self {
                    #zeroed_expr
                }
            }

            fn out_with_layers(&self, input: Self::InputType) -> Self::Layers {
                #layer_exprs
                Self::Layers {
                    #layer_exprs_fields
                }
            }

            fn backprop(
                &self,
                input: Self::InputType,
                grad: &mut Self,
                err: <Self::Layers as goober::OutputLayer>::Type,
                layers: Self::Layers,
            ) {
                #backprop_exprs
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn gen_add_impl(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote!(self.#name += rhs.#name;)
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_layer_fields(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        let ty = &f.ty;
                        quote!(#name: <#ty as goober::OutputLayer>::Type,)
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_output_type(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let f1 = fields.named.last().unwrap();
                    let name = &f1.ident;
                    let ty = &f1.ty;
                    quote! {
                        type Type = <#ty as goober::OutputLayer>::Type;
                        fn output_layer(&self) -> Self::Type {
                            self.#name
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_input_type(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let f1 = fields.named.first().unwrap();
                    let ty = &f1.ty;
                    quote!(<#ty as goober::InputLayer>::Type)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_adam_expr(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote!(self.#name.adam(&g.#name, &mut m.#name, &mut v.#name, adj, lr);)
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_zeroed_expr(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        let ty = &f.ty;
                        quote!(#name: <#ty>::zeroed(),)
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_layer_exprs(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let mut prev = &None;
                    let recurse = fields.named.iter().enumerate().map(|(i, f)| {
                        let name = &f.ident;
                        let res = if i > 0 {
                            quote!(let #name = self.#name.out(#prev);)
                        } else {
                            quote!(let #name = self.#name.out(input);)
                        };

                        prev = name;
                        res
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_layer_exprs_fields(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote!(#name,)
                    });
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}

fn gen_backprop_exprs(data: &Data) -> TokenStream {
    match *data {
        Data::Struct(ref data) => {
            match data.fields {
                Fields::Named(ref fields) => {
                    let mut prev = &None;
                    let mut list = fields.named.iter().enumerate().map(|(i, f)| {
                        let name = &f.ident;
                        let res = if i > 0 {
                            quote!(let err = self.#name.backprop(&mut grad.#name, err, layers.#prev, layers.#name);)
                        } else {
                            quote!(let err = self.#name.backprop(&mut grad.#name, err, input, layers.#name);)
                        };

                        prev = name;
                        res
                    }).collect::<Vec<TokenStream>>();
                    list.reverse();
                    let recurse = list.into_iter();
                    quote!(#(#recurse)*)
                }
                _ => unimplemented!(),
            }
        }
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}
