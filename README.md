An idea I had listening around conversations at my birthday: can I compute the vibe of a room? What does that even mean? The idea to use a Hodge decomposition immediately came.

I'm curious to see what will happen.

Idea: Split 1 audio file into 3, corresponding to the 3 pieces of the Hodge decomposition. The harmonic part is the "vibe" of the room. The other two are because I'm curious and they came for free. There are 3 more files, corresponding to the pairwise combos, which are

1.  the "vibe" of the room
2.  the curl of the room
3.  the divergence of the room

TODO elaborate hodge and how helmholtz works too in this case.


Use torchaudio to load 1 audio file, call it `input`. `input` should be turned
into a (possibly complex) vector field. Elaborate how this is done, then implement
it.


Elaborate how the Helmholtz or Hodge decomposition works, then implement it.
