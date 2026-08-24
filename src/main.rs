pub const LIBRARY_NAME: &'static str = "computer_vision_rust";

fn main() {
    println!("Hello, world!");

    let mut cmd_args = std::env::args();
    if cmd_args.len() >= 2 { // Test mode
        assert!(cmd_args.len() >= 3, "Correct usage is: cargo run test [<sct-name>]");

        let _exec_path = cmd_args.next().unwrap();
        assert_eq!(cmd_args.next().unwrap(), "test", "Forgot \"test\".");

        let sct_name = cmd_args.next().unwrap();
        computer_vision_rust::test::run_sct(&sct_name);
    }

    
}
