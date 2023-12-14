use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Data, DeriveInput, Field, Fields};

#[proc_macro_derive(FeedForwardNetwork)]
pub fn network_utils(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let layer_name = Ident::new((name.to_string() + "Layer").as_str(), Span::call_site());

    let first = first_field(&input.data);
    let first_name = &first.ident;
    let first_type = &first.ty;

    let add_impl = gen_add_impl(&input.data);
    let layer_fields = gen_layer_fields(&input.data);
    let output_type = gen_output_type(&input.data);
    let input_type = gen_input_type(&input.data);
    let adam_expr = gen_adam_expr(&input.data);
    let layer_exprs = gen_layer_exprs(&input.data);
    let layer_exprs_fields = gen_layer_exprs_fields(&input.data);
    let backprop_exprs = gen_backprop_exprs(&input.data);

    let sparse = if first_type.to_token_stream().to_string().contains("Sparse") {
        let first_output_type = quote!(<<#first_type as goober::FeedForwardNetwork>::Layers as goober::OutputLayer>::Type);
        let layer_exprs_sans = gen_layer_exprs_sans(&input.data);

        quote! {
            impl #name {
                pub fn add_feature(&self, accumulator: &mut #first_output_type, idx: usize) {
                    *accumulator += self.#first_name.weights_row(idx);
                }

                pub fn sub_feature(&self, accumulator: &mut #first_output_type, idx: usize) {
                    *accumulator -= self.#first_name.weights_row(idx);
                }

                pub fn out_from_first(&self, #first_name: &#first_output_type) -> <<Self as goober::FeedForwardNetwork>::Layers as goober::OutputLayer>::Type {
                    #layer_exprs_sans
                }
            }
        }
    } else {
        quote!()
    };

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

            fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
                use goober::OutputLayer as __InternalOutputLayer;
                #layer_exprs
                Self::Layers {
                    #layer_exprs_fields
                }
            }

            fn backprop(
                &self,
                input: &Self::InputType,
                grad: &mut Self,
                err: <Self::Layers as goober::OutputLayer>::Type,
                layers: &Self::Layers,
            ) -> Self::InputType {
                use goober::OutputLayer as __InternalOutputLayer;
                #backprop_exprs
            }
        }

        #sparse
    };

    proc_macro::TokenStream::from(expanded)
}

macro_rules! struct_with_fields_only {
    (|$data:ident, $fields:ident| $thing:expr) => {{
        match *$data {
            Data::Struct(ref $data) => match $data.$fields {
                Fields::Named(ref $fields) => {
                    $thing
                },
                _ => unimplemented!(),
            },
            Data::Enum(_) | Data::Union(_) => unimplemented!(),
        }
    }};
}

fn first_field(data: &Data) -> &Field {
    struct_with_fields_only!(|data, fields| fields.named.first().unwrap())
}

fn gen_add_impl(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let recurse = fields.named.iter().map(|f| {
            let name = &f.ident;
            quote!(self.#name += &rhs.#name;)
        });
        quote!(#(#recurse)*)
    })
}

fn gen_layer_fields(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let recurse = fields.named.iter().map(|f| {
            let name = &f.ident;
            let ty = &f.ty;
            quote!(#name: <#ty as goober::FeedForwardNetwork>::Layers,)
        });
        quote!(#(#recurse)*)
    })
}

fn gen_output_type(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let f1 = fields.named.last().unwrap();
        let name = &f1.ident;
        let ty = &f1.ty;
        quote! {
            type Type = <<#ty as goober::FeedForwardNetwork>::Layers as goober::OutputLayer>::Type;
            fn output_layer(&self) -> Self::Type {
                self.#name.output_layer()
            }
        }
    })
}

fn gen_input_type(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let f1 = fields.named.first().unwrap();
        let ty = &f1.ty;
        quote!(<#ty as goober::FeedForwardNetwork>::InputType)
    })
}

fn gen_adam_expr(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let recurse = fields.named.iter().map(|f| {
            let name = &f.ident;
            quote!(self.#name.adam(&g.#name, &mut m.#name, &mut v.#name, adj, lr);)
        });
        quote!(#(#recurse)*)
    })
}

fn gen_layer_exprs(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let mut prev = &None;
        let recurse = fields.named.iter().enumerate().map(|(i, f)| {
            let name = &f.ident;
            let res = if i > 0 {
                quote!(let #name = self.#name.out_with_layers(&#prev.output_layer());)
            } else {
                quote!(let #name = self.#name.out_with_layers(input);)
            };
            prev = name;
            res
        });
        quote!(#(#recurse)*)
    })
}

fn gen_layer_exprs_fields(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let recurse = fields.named.iter().map(|f| {
            let name = &f.ident;
            quote!(#name,)
        });
        quote!(#(#recurse)*)
    })
}

fn gen_backprop_exprs(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let mut prev = &None;
        let mut list = fields.named.iter().enumerate().map(|(i, f)| {
                let name = &f.ident;
                let res = if i > 0 {
                    quote!(let err = self.#name.backprop(&layers.#prev.output_layer(), &mut grad.#name, err, &layers.#name);)
                } else {
                    quote!(self.#name.backprop(input, &mut grad.#name, err, &layers.#name))
                };

                prev = name;
                res
            }).collect::<Vec<TokenStream>>();
        list.reverse();
        let recurse = list.into_iter();
        quote!(#(#recurse)*)
    })
}

fn gen_layer_exprs_sans(data: &Data) -> TokenStream {
    struct_with_fields_only!(|data, fields| {
        let mut prev = &fields.named.first().unwrap().ident;
        if fields.named.len() > 1 {
            let recurse = fields.named.iter().skip(1).map(|f| {
                let name = &f.ident;
                let res = quote!(let #name = self.#name.out(&#prev););

                prev = name;
                res
            });
            quote! {
                #(#recurse)*
                #prev
            }
        } else {
            quote!(*#prev)
        }
    })
}
