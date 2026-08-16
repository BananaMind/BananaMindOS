# Contributing

Run `make check` and `make kernel` before submitting changes. Changes to model
layouts should also be validated with `tools/bm2n_info.py`, the host runner,
and at least one QEMU boot using `-cpu 486`.

Do not commit `build/`, downloaded checkpoints, converted BM2NQ models, or ISO
images. Keep model URLs revision-pinned so builds do not silently change.
