# tiny_corpus

A handful of short, public-domain U.S. founding-era documents used as a
fixture for the `.kb` packer/reader demo and for `tests/test_retrieval.py`.

Files:

- `federalist_10_factions.txt` — Madison on the nature of factions
- `federalist_10_property.txt` — Madison on property and the latent causes of faction
- `federalist_51_checks.txt` — Madison on the separation of powers
- `gettysburg.txt` — Lincoln, Gettysburg Address (1863)

All sources are public domain. The default chunker turns these into
roughly 12–20 ~500-character chunks — enough to make top-3 retrieval
non-trivial but small enough to pack and query in seconds.
