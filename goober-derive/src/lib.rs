use proc_macro::TokenStream;

fn get_struct_info(item: TokenStream) -> Option<(String, Vec<(String, String)>)> {
    let mut fields = Vec::new();
    let mut name = None;
    let mut prev = String::new();
    let mut cache = String::new();

    for token in item {
        let token_str = token.to_string();

        if prev == "struct" {
            name = Some(token_str.clone());
        }

        if !cache.is_empty() {
            fields.push((cache.clone(), token_str.clone()));
        }

        cache = if token_str == ":" {prev} else {String::new()};
        prev = token_str;
    }

    name.map(|name| (name, fields))
}

#[proc_macro_derive(NetworkUtils)]
pub fn derive_network_utils(item: TokenStream) -> TokenStream {
    let (name, fields) = get_struct_info(item).unwrap();
    println!("{name}");
    println!("{fields:#?}");

    let mut block = String::new();

    block.push_str(format!("pub struct {name}Layers {{").as_str());

    for (field, field_type) in &fields {
        block.push_str(format!("{field}: {field_type},").as_str());
    }

    block.push('}');



    block.parse().unwrap()
}
