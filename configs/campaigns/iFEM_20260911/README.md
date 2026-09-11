# iFEM native AI-CAE4ALL campaign

Three primary candidates use the native AI-CAE4ALL trainer and `.pth` checkpoints. Run from this suite root; config paths are deliberately relative to each method cwd.
The 8,809-case train HDF5 is native-split into 7,047 / 880 / 882 with seed 42.
The 979-case infer HDF5 is held out from training. MGN alone uses a private train HDF5 copy
because its native trainer writes normalization metadata.

Stage 1 adds one controlled arm per model: T02 changes only Transolver LR to 1e-3,
M03 changes only MeshGraphNets LR to 3e-4, and D03 changes only DeepONet weight
decay to 1e-4. GPU 0 remains unused. Geometric augmentation stays disabled because
this iFEM target contains two 2-D vector fields and 12 condition channels are also
directional; the native generic augmentation does not transform that contract.
