# SOTA comparison: checkpoint and implementation registry (2026-11-02)

**Purpose.** Records exactly which weights and code would be evaluated. Required by protocol §8, before any inference.

**Storage.** Downloaded assets live **outside the repository**, in `D:\MARINE_NET_RESEARCH_sota_assets\`, so large files never enter git. Nothing is committed.

**Hashing.** All hashes are SHA-256 of the file bytes, computed after download.

## Group R: released checkpoints (authors' weights; evaluated as published)

| Method | File (under `checkpoints\`) | Bytes | SHA-256 | Source | Licence | Training exposure |
|---|---|---|---|---|---|---|
| **WaterNet** | `waternet_exported_state_dict-daa0ee.pt` | 4,373,375 | `daa0eeb74271c402b6772843a61dc751fcb243553e6650891285ad29ce7c7bc9` | Dropbox link in tnwei/waternet README (the port's default weights) | port: MIT. **The weights are the original authors' weights** (Li-Chongyi/Water-Net_Code, no licence): not redistributed | random 800 UIEB (read-directly) |
| **U-Shape Transformer** | `UShape_saved_models.zip` → `UShape_extracted\saved_models\G\generator_795.pth` | 117,249,306 (zip); 126,569,949 (pth) | zip `32df84c63043aa1522d6b55640d56ab296d78733b7700ad777a27d1a76e696e2`; pth `c595c2aa49cd9623d02379b3990ad955c46d65dd3095b2b1c5612744981d8fae` | Google Drive `19a_kDJTT5S96kzwQntEMhSxAPYw4xY2P` (official README) | MIT | LSUI |
| **WWE-UIE** | `WWE-UIE_best_model.pth` | 3,046,764 | `36c3e59e556ed69a31e3f83ae7e15b86409eeb951de049ebc2207728c210027d` | Google Drive `1c80kkDbgpLD1MtqVLql1Y1NQVD7IzxiO` (official README, "[UIEB]") | Apache-2.0 | UIEB |
| **FUnIE-GAN** (official TF-Keras) | `model_15320_.h5` + `model_15320_.json` | 16,940,688 + 14,977 | h5 `3054abfc7b77706b99fd9a2e92ed8df86f6e041ac2af3d6f4a4d52292b974782`; json `cbbfe3b117a81d74b6382f01cb700453e739a319a3a6f38b77419bc9a1e6cf83` | in-repository `TF-Keras/models/gen_p/` at commit `8f934c834c94e007b00866186b9ee624dc2b7b69` (git blob `78a6ce33…`) | MIT (LICENSE text) | EUVP paired |
| **NU²Net** | `NU2Net_ckpt.pth` | 37,773,771 | `2e1401a70e88d7dbc37bd77f709ff74ec52cc30b6d8a698b4cdf8842a15821d5` | Google Drive `1EHKNW3dpLeR7Y1ayP7sZ2uikMGOjyC8y` (official folder) | **CC BY-NC 4.0** | URankerSet / UIEB random 800 |

**Integrity and provenance evidence:**

| Method | Evidence |
|---|---|
| WaterNet | the SHA-256 begins with `daa0ee`, the hash prefix in the published filename |
| FUnIE-GAN | the h5 size equals the GitHub API blob size |
| NU²Net | internal metadata: epoch 237, `max_psnr` 22.668749761892308, `max_ssim` 0.9246348440284826. This equals the paper's Table 2 row **with** URanker loss (22.669 / 0.9246), so the checkpoint is that variant |

**Code used to construct the networks:**

| Method | Code |
|---|---|
| WaterNet | `models/waternet_wrap.py` (tnwei/waternet @ `2091896881fe…`, already pinned in the project) |
| WWE-UIE | `models/wwe_uie.py` (WWE-UIE @ `fd0697558203…`, pinned) |
| U-Shape | `D:\MARINE_NET_RESEARCH_sota_assets\repos\U-shape_Transformer_for_Underwater_Image_Enhancement` @ `41f9897669d2e6f6718e00772ece2523d0f149a2`, `net/Ushape_Trans.py: Generator()` |
| NU²Net | `…\repos\UnderwaterRanker` @ `f9bf3508121c5bd658a2f2c3fa8d4d3826ed1ce0`, `all_model/NU2Net/model.py: NU2Net(basic_channel=64)` (`options/NU2Net.yaml`) |
| FUnIE-GAN | `…\repos\FUnIE-GAN` @ `8f934c83…`, `tf_keras.models.model_from_json` + `load_weights`. Only `TF-Keras/`, `PyTorch/`, `LICENSE` and `README.md` were checked out: `data/con.jpg` is a reserved filename on Windows |

### Classical methods (no weights)

| Method | Code | Status |
|---|---|---|
| CLAHE | `v2_tools/sota_classical.py: clahe()` (OpenCV 5.0.0; Lab-L, clip 2.0, tiles 8×8) | tests pass |
| UDCP | `v2_tools/sota_classical.py: udcp()`, implemented from Drews et al. 2013 | tests pass; the guided-filter radius awaits a decision |
| IBLA | — | not used |

## Group T: retrained under our LSUI recipe (architecture comparison; never mixed with Group R)

| Model | Run | Checkpoint SHA-256 (best_valA, verified against `result.json` by `paper1_final_eval_v2.verify_checkpoint2`) | Seeds |
|---|---|---|---|
| WaterNet (LSUI retrain) | `runs_v2/waternet_lsui_s0` | `af168d31cf1632bb2f9c550e32670eab1c39f28714c2a60bdfedbfc30c3565b3` | 1 |
| WWE-UIE (LSUI retrain) | `runs_v2/wwe_lr2e-4_s0` | `fc9ebbeaffb1f56ca4297ef5d9a5b914274e5d6db32845cfd067fad490ca4bf5` | 1 |
| MARINE-Net / AT-Trunk, BN / IN | the 12 Stage 4 checkpoints | as in `TUB_stage5_implementation_freeze_addendum.md` §2 | 3 each |

## Environment

- **Isolated venv:** `D:\MARINE_NET_RESEARCH_sota_assets\venv_sota`.
- **Build:** Python 3.12, `--system-site-packages`.
- **Local packages:** only `wrapt==2.5.2`. This was installed because TensorFlow 2.21 failed to import with the system's `wrapt` 1.12.1 (`formatargspec` removed in Python 3.12). The system `wrapt` is unchanged.
- **Versions:**
  - torch 2.7.1+cu118;
  - TensorFlow 2.21.0 and tf_keras 2.21.0. TensorFlow runs on **CPU** here (no Windows GPU build for TensorFlow ≥ 2.11);
  - OpenCV 5.0.0.

**Safe loading.** Checkpoints are loaded with `torch.load(weights_only=True)`. For NU²Net only, the NumPy scalar and dtype types it contains are allow-listed by explicit name (`numpy.core.multiarray.scalar`, `numpy.dtypes.Float64DType`). `weights_only=False` is never used.


## Status update (2026-11-02, Amendment 02; appended, nothing above edited)

- **WaterNet (original authors' weights, `daa0eeb7…`): EXCLUDED from Group R.** The original repository has no licence and no documented permission was obtained, so research-use permission is not assumed. The file stays on disk, unused. WaterNet appears only as our LSUI retrain in Group T (exploratory, n = 1).
- **Group R in use:** U-Shape, WWE-UIE (official), FUnIE-GAN (official Keras), NU²Net, CLAHE, UDCP.
