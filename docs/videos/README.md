# Studio video walkthroughs

These videos were recorded on October 8, 2026 against the local AI-CAE4ALL Studio.
They show actual repository data and browser interactions. English captions are
included in the MP4s, with separate SRT files for accessible playback and reuse.

Play either recording directly below. Both are 2:39 at normal speed.

## DeepJEB workflow

https://github.com/user-attachments/assets/76410429-83e7-4492-aa9f-42115051c48e

[English subtitles](deepjeb-workflow.en.srt) | [Original MP4](deepjeb-workflow.mp4)

The recording starts with the six core pipeline blocks and clicks the zoom-in
button three times. It pans to each model, opens its inspector, chooses a generated
bracket, rotates the mesh, and clicks the stress and vertical-displacement
channels. It then adds vertical deflection as a second optimization objective,
sets a 0.4 mm viewing constraint, evaluates the candidate CSV, and clicks a
Pareto point to reopen that saved design. The entire take is continuous at normal
speed, with deliberate reading pauses and visible cursor/click feedback.

The presentation uses a separate browser context and hides optional metric and
held-out evaluation branches. The recorded candidate table is a completed
100-design screening run. With the illustrated mass/vertical-deflection
objectives and 0.4 mm bound, 76 designs are feasible and 12 are Pareto-optimal;
five are selected. These counts describe this saved candidate population and
this demonstration constraint.

## Training walkthrough

https://github.com/user-attachments/assets/72b5dfa4-2f81-4ee2-a1af-0ce5f0ca6796

[English subtitles](deepjeb-from-scratch.en.srt) | [Original MP4](deepjeb-from-scratch.mp4)

The recording starts with an empty canvas, inspects an SDF dataset, adds SDFFlow,
enters new checkpoint/output paths, connects the dataset, and clicks Train.
The native SDF-VAE training loop starts. The demonstration job is then stopped;
one labelled 0.4-second transition skips the training wait and switches to
previously trained SDFFlow and HI-MGN models. The remaining walkthrough browses
available model families and inspects existing candidate fields and Pareto results.
The fresh demonstration job did not produce those pretrained results.

## Data and interpretation

The local recordings use `ex1_deepjeb.h5` for geometry learning,
`ex13_deepjeb_ver.h5` for the HI-MGN structural model, and the completed run's
`screening.csv` and `designs.h5` for candidate inspection. The fields shown are
HI-MGN predictions. The workflow is screening (`opt_budget = 0`), with an
illustrative 0.4 mm constraint; it does not demonstrate an independently
FEA-verified optimum or an empirical ranking of model families.

Datasets, model weights, and generated candidates remain local and are not
distributed with these recordings. To reproduce the workflow with your own
artifacts, prepare compatible datasets, train or supply compatible checkpoints,
configure the CAD Generator, then connect its candidate table to Optimization.
The checked-in native optimization config uses FEA with a positive search budget,
so adjust its backend and budget for a surrogate screening workflow.

`manifest.json` records file sizes, SHA-256 checksums, inline attachment URLs,
and playback metadata. The GitHub attachments contain the same full-quality
files as the originals in this directory.
The two MP4s are curated documentation assets. Local raw takes, job logs,
checkpoints, and alternate edits remain outside the published documentation.

## Further demonstrations

- Compare MeshGraphNets, HI-MGN, and Transolver on compatible held-out cases,
  using the same target channels, units, split, and metric definitions.
- Compare surrogate fields with independent FEA on the same generated geometry,
  showing error fields and measured runtimes together.
- Change the deflection bound on one fixed candidate population and show how
  feasible designs, the Pareto set, and selected shapes change.

These are proposed recordings. Their results need matched experiments before
publishing a ranking or an accuracy claim.
