An idea I had listening around conversations at my birthday: can I compute the vibe of a room? What does that even mean? The idea to use a Hodge decomposition immediately came.

I'm curious to see what will happen.

Idea: Split 1 audio file into 3, corresponding to the 3 pieces of the Hodge decomposition. The harmonic part is the "vibe" of the room. The other two are because I'm curious and they came for free. There are 3 more files, corresponding to the pairwise combos, which are

1.  the "vibe" of the room
2.  the curl of the room
3.  the divergence of the room

TODO elaborate hodge and how helmholtz works too in this case.

If Hodge is too hard, use Helmholtz and included the poloidal and toroidal split.

Use torchaudio to load 1 audio file, call it `input`. It will have 2 channels and thousands of samples, with a shape of `2,?`. How to turn the `input` complex dtpye tensor into a complex vector field? Elaborate how this is done, then implement it in Python.

Elaborate how the Helmholtz or Hodge decomposition works, then implement it.

Account for the negative-definite signature of the time dimension, because it *is* timelike.

Implement all this in Python, using Python 3.10+. Include type annotations,
pathlib, and other modern niceties.
