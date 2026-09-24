# Public test fixtures

Shipped whole with the public export (and gated like everything else).

| file | what | provenance |
|---|---|---|
| `dose_bands/w5_r32_measured.json` | a judged dose band (3B w5 rank-32 map) | banked measurement; numbers verbatim, provenance strings scrubbed |
| `dose_bands/residual_scale_3b_8b.json` | per-site residual norms, 3B vs 8B | banked measurement; numbers verbatim, bank paths scrubbed to basenames |
| `length_null_sitenorm_8b.json` | reply/candidate lengths + judged ranks from one 8B run | banked measurement; source paths scrubbed to basenames |

The numbers are real measurements, kept where a synthetic stand-in would test
nothing. They are regenerated from the private originals by a script in the
private tree (it checks each output against the export gate); a private test
fails if a committed fixture drifts from its original. Everything else the
public suite needs is built in-test from seeded synthetic data.
