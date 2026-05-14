// I/O layer — Phase 3
pub mod fasta;
pub mod output;
pub mod parquet;
pub mod tsv;

use std::path::Path;

/// Detected input file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Fasta,
    Tsv,
    Csv,
    Parquet,
}

/// Detect the input format from the file extension.
///
/// Returns `None` if the extension is unrecognised.
pub fn detect_format(path: &Path) -> Option<InputFormat> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "fasta" | "fa" | "fna" | "fas" => Some(InputFormat::Fasta),
        "tsv" | "txt" => Some(InputFormat::Tsv),
        "csv" => Some(InputFormat::Csv),
        "parquet" | "pq" => Some(InputFormat::Parquet),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_detect_format() {
        assert_eq!(
            detect_format(Path::new("seqs.fasta")),
            Some(InputFormat::Fasta)
        );
        assert_eq!(
            detect_format(Path::new("seqs.fa")),
            Some(InputFormat::Fasta)
        );
        assert_eq!(detect_format(Path::new("seqs.tsv")), Some(InputFormat::Tsv));
        assert_eq!(detect_format(Path::new("seqs.csv")), Some(InputFormat::Csv));
        assert_eq!(
            detect_format(Path::new("seqs.parquet")),
            Some(InputFormat::Parquet)
        );
        assert_eq!(
            detect_format(Path::new("seqs.pq")),
            Some(InputFormat::Parquet)
        );
        assert_eq!(detect_format(Path::new("seqs.json")), None);
        assert_eq!(detect_format(Path::new("no_extension")), None);
    }
}
