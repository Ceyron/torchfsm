TorchFSM 0.0.5 Release Notes

## Feature Addition
* Add `Dispersion`, `HyperDiffusion`, `Leray`, `ChannelWisedDiffusion`, `GrayScottSource` operators.
* Add `KPPFisher`, `SwiftHohenberg`, `GrayScott` equations.
* Add `random_gaussian_blobs`, `truncated_fourier_series_customed_filter`, `random_truncated_fourier_series` field function.
* Add `plot_traj_frames` and `plot_3d_traj_slices` plot functions.
* Add error notation (ce886355c7f416568ed6a2870d52eb891a3ab50a).
* Add `normalized_low_pass_filter` function for mesh (5f945bcb9d635905095f51fd6f828402a4df98d7).
* Allow to check nan values during integrations (89669aa86df632292d485d5881a1e70e8096ebc9,89669aa86df632292d485d5881a1e70e8096ebc9).
* Introduce `normalize_mode` to fields (ab0b899fe11e3d65be54a48c63883c6b28fdab11).
* Allow to initialize operator directly from `LinearCoef` and `NonlinearFunc` (8abc939dd5691810026038d1991994f42da8bda0).
* Allow to return frame indices for fram select functions (adfb4cae620095721aa763f55aa1cf398237e574).
* Allow real time ifft for `CPURecorder` (14256f56c097cd8a54a1cfb45ca555c52777ee88,8ba561a9419bcc3df814499c4418c60731589eee).
* Allow to stop simulation from recorder (8ba561a9419bcc3df814499c4418c60731589eee).

## Peformance Improvement
* Change `Convection` operator to a more memory and speed efficient version (2e996775e356e21945c2a13467a9c922055ea8ef).
* Remove lru cache to avoid memory leakage when run the simulation multiple times (f1f55b356bb461b6f47dd38927127046540871f1)

## Feature Change
* Modify the logic for recording the initial conditions ( 494d74a9232f656ea2b80602f29c3381e525e438)
* Update `KdV` equation with `Dispersion` operator to suppot high-dimensional simulation (1ff30c0f52b555f06031e5f143597af8a68a3708).
* Make saved plot tight layout by default (5dc9ecb0a2a82908c6ea144a8062baa76bdad2be).

## Bugs Fixing
* Fix error raising issue of `_KSConvectionGenerator` (77fcb0387624095f2612e9da87a65ee9ddf2668d) 
* Fix None dict error when setting `de_aliasing_rate` (978a103328a654f77d355d5a2c2c6a3a0f1c36f1).
* Fix type convert issue in traj postprocess funcs (5e4757597b65102c7764ab64822f518a62b14543).
* Fix cmap error for rendering 3d field (917653ac15676dbec13f2cfeecf273308787ca4a)

Update by @qiauil 

TorchFSM 0.0.4 Release Notes
## Performance Improvement
* Add garbage clean to optimize memory usage in 5815dc09d46a39b39b4e5b46387e32cd895caf08
* Make LR in ETDRK integrator a non-attribute variable to optimize memory performance in 74744e77c13b6390cea56e4ccbaa4e66cd8dc0a0
* Add standard ETDRK integrator and rename the original integrator as SETDRK; Allow CPU cache when building the SETDRK integrator to save memory in bb2612a71e38920654b2fd71d1d1664d116b17ce

## Functional Update
* Make normalization of `diffused_noise` batch-wised in cbde32fab6622b7cdd3a3b4ff3dec9faa2e1e46b
* Make `unit_variance` and `unit_magnitude` exclusive in diffused_noise in aeace8d092e50fc2188d8364655c6f74871745ce
* Add `RandomBatchWisedRecorder` in 6d49ff36db8a79da958cdfba89b568cc479c1e0f
* Add `print_gpu_memory function` in 453da36866d1395bf197999d736ff95a77a41ae2
* Add an absolute low-pass filter for operators in e62f827f75d1e27e6628c481a746d0f1a0874632
* Add `truncated_fourier_series` as a field function in 050fbf2925ccdcee692a361cafbe88fb2c87836d

## Name Changes
* Typo fix in f0d6fd8ace001100bc355256df43b3747c95ad45
* Change parameter name `n_batch` to `batch_size` in d5d8f779bc51bdf6c7aa56a0bf1b2d8c01da9584
* Change parameter name `num_circle_point` to `n_integration_points` and `circle_radius` to `integration_radius` for ETDRK intergrators in 0f1497765a1922f79c2861cd3aa1af9129f513c0

Update by @qiauil 