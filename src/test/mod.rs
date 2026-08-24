use serde::{Serialize, Deserialize};

/// The path to the sct-lists json file.
pub const SCT_LIST_PATH: &'static str = "src/test/sct-list.jsonc";

#[derive(Serialize, Deserialize)]
pub struct Sct {
    pub name: String,
    pub path: String,
    pub options: Vec<String>,
}
#[derive(Serialize, Deserialize)]
pub struct SctListFile {
    #[serde(rename = "sct-list")]
    pub sct_list: Vec<Sct>,
}

/// Tries to run the sct with the specified `name`.
pub fn run_sct(name: &String) {
    let sct_list_file = std::fs::read_to_string(SCT_LIST_PATH);
    assert!(sct_list_file.is_ok(), "Could not read sct-list file contents due to error: {:?}.", sct_list_file.err());

    let json_data: Result<SctListFile, serde_json::Error> = serde_json::from_str(sct_list_file.unwrap().as_ref());
    assert!(json_data.is_ok(), "Could not parse sct-list json data due to error: {:?}.", json_data.err());

    let data = json_data.unwrap();

    let sct = data.sct_list.iter().find(|sct| sct.name == *name);
    assert!(sct.is_some(), "Could not find sct with name: {:?} in sct-list.", *name);

    println!("Running \"cargo run --bin {:?} {:?}\".", name, String::from_iter(sct.unwrap().options.iter().map(|s| (*s).clone())));

    if let Err(exec_err) = std::process::Command::new("cargo")
        .arg("run")
        .arg("--bin")
        .arg(&sct.unwrap().name)
        .args(sct.unwrap().options.iter())
        .spawn()
    {
        panic!(
            "Running \"cargo run --bin {:?} {:?}\" failed with error: {:?}.",
            name,
            String::from_iter(sct.unwrap().options.iter().map(|s| (*s).clone())),
            exec_err
        );
    }

}
