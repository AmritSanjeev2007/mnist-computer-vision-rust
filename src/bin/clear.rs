
const OUT_PATH: &'static str = "tests/out";


fn main() {
    println!("Clearing content at {:?} directory...", OUT_PATH);

    'remove_dirs: {
        if let Err(err) = std::fs::remove_dir_all(std::path::Path::new(OUT_PATH)) {
            println!("Could not remove content at {:?} due to error: {:?}.", OUT_PATH, err);
            break 'remove_dirs;
        }
    }

    'create_dirs: {
        if let Err(err) = std::fs::create_dir(std::path::Path::new(OUT_PATH)) {
            println!("Could not create directory at {:?} due to error: {:?}.", OUT_PATH, err);
            break 'create_dirs;
        }
    }

    println!("Finished clearing content and remaking directories.");
}