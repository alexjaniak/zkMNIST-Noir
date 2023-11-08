use dep::std;

global N = 30;
fn main(inputs: [Field; N]) -> pub [Field; 2] {
    std::hash::pedersen(inputs)
}