"""Model loading and decoder-structure helpers.

- :mod:`pleroma.model.load` — ``DTYPES``, ``load_frozen_model`` (HF, frozen,
  eval, SDPA), ``resolve_pad_token_id``. Imports torch at module scope.
- :mod:`pleroma.model.layers` — ``decoder_layers``, ``decoder_stack``,
  ``model_hidden_dim``: wrapper-aware lookups, torch-free.

Submodules are imported explicitly so this package does not pull torch in on
import.
"""
