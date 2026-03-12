fn main() {
    let args: Vec<String> = std::env::args().collect();
    match ignition::cli_runner::run_cli(args) {
        Ok(()) => {}
        Err(e) => {
            // clap already printed its own error message; only emit ours for
            // non-clap (i.e. actual runtime) errors.
            let msg = e.to_string();
            if !msg.contains("clap") && !msg.contains("Usage:") {
                eprintln!("ignition: error: {e}");
            }
            std::process::exit(1);
        }
    }
}
