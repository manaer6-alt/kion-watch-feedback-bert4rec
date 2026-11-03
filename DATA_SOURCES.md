# Data, code and scope

Dataset: Petrov A., Safilo I., Tikhonovich D., Ignatov D. MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation. 2022. https://arxiv.org/abs/2209.00325

Download origin: https://github.com/irsafilo/KION_DATASET at revision `f69775be31fa5779907cf0a92ddedb70037fb5ae`. The loader verifies the archive SHA-256 before parsing. This repository does not redistribute the raw archive or assign it a new licence. Obtain it from the official source under its applicable terms. The MIT licence covers our code; third-party datasets, packages and cited publications remain under their own terms.

The `watched_pct` field is an aggregate across views of a user-item pair. Its row uses the last interaction date. Only `content_type == 'film'` is selected; missing percentages are excluded consistently and zero is a valid observation. All source interactions are unique user-item aggregates; within-day input order is item ID. No claim is made about individual playback sessions or the time at which a percentage threshold was first crossed.

STEP10 copies the *form of the mapping* in RecTools' official Transformer Models Advanced Training Guide: `np.where(watched_pct > 10, 3, 1)`. Our experiment applies it in a shared normalised masked-target loss and does not purport to reproduce RecTools' training results. Reference: https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_advanced_training_guide.html (checked 2026-10-31).

The reported scientific implementation is preserved unchanged with SHA-256 `5ab07ac369ae3966f3d781ada94a38415eed710ade65541c594c2a4fd3b57509`. New wrappers, independent result verification and plotting do not change its numerical training logic. The notebook embeds the same source. Recorded plan hash: `8f922f1b9331d510213a000134faf72aae7c50ebeffcd1df41ad03dd7649d4f5`.

`reference_results` contains aggregate metrics, learning curves, environment and hash manifests, and the already completed independent numerical audit. It contains no per-user recommendation tables and no checkpoint weights. A new full run generates those tables locally, allowing independent rechecking. The supplied audit does not establish that the original checkpoint contents match independently retrained models.
