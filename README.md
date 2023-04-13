An idea I had listening around conversations at my birthday: can I compute the vibe of a room? What does that even mean? The idea to use a Hodge decomposition immediately came.

Hodge splits a vector field into 3: divergence free (conservative), curl free, and harmonic (irrotational and incompressible).

I'm curious to see what will happen.

Idea: Split 1 audio file into 3, corresponding to the 3 pieces of the Helmholtz decomposition. The harmonic part is the "vibe" of the room. The other two are because I'm curious and they came for free. There are 3 more files, corresponding to the pairwise combos curl+ div, harmonic + curl, and harmonic + div

Use Python 3.10+, type annotations, argparse, and Pathlib for file handling. Use `torchaudio` and PyTorch more generally to load 1 audio file, call it `input`. It will have 2 channels and thousands of samples, for a shape of `2,?`. The `2` refers to left and right channels. Convert it to a power spectrogram (ideally you would work with the complex vector field, but that may be too hard). Turn that into a vector field. Account for the multiple channels.

Elaborate how the Helmholtz or Hodge decomposition works, then implement it. If Hodge is too hard, use Helmholtz and include the poloidal/toroidal split.

Output the code as a single block, with comments. The code should be self-contained, and not require any external files. You may assume any useful libraries.

## How to run

Input is a WAV file, output is a directory, containing WAV files for each piece.

```bash
poetry install
poetry run python main.py $INPUT.wav  $OUTPUT_DIR
```
