# Vendored third-party code

## saicinpainting (LaMa)

- Upstream: https://github.com/advimman/lama
- Commit: `786f5936b27fb3dacd2b1ad799e4de968ea697e7` (2025-02-05)
- Licence: Apache License 2.0, Samsung Research
- Weights: `big-lama.zip`, from the URL the upstream README itself gives
  (`https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip`),
  sha256 `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`

### What is here and why

The edit stack needs one thing from LaMa: the `FFCResNetGenerator` definition, so
the pinned `big-lama` checkpoint can be loaded and run. Installing the upstream
package would pull `pytorch-lightning`, `hydra-core`, `kornia` and `webdataset`
into an environment built around torch 2.14 and numpy 2.5, for code paths that
only matter during training. Vendoring the generator's import closure is six
files and about 28 KB.

Verbatim, at the commit above:

- `training/modules/ffc.py`
- `training/modules/base.py`
- `training/modules/spatial_transform.py`
- `training/modules/squeeze_excitation.py`
- `training/modules/depthwise_sep_conv.py`
- `training/modules/multidilated_conv.py`

Reduced, and marked as such in its own docstring:

- `utils.py` — only `get_shape`, the single symbol `ffc.py` imports from it.
  Upstream's version imports `pytorch_lightning` at module level.

Nothing here is modified otherwise. The loader that uses it is
`idea91/edits/lama.py`; the licence row is in `data/LICENSES.md` and the pins in
`shared/env/PINS.md`.
