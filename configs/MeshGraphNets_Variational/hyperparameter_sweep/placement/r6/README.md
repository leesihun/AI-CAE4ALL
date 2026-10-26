# Placement sweep r6 (generated)

See ../README.md for the design. 98 jobs (89 trainings, 9 sampling only) on 14 slots; estimated makespan 16 days (guess: 34 h per 120 epochs on one card, x1.5 for two per card, width^1.5, batch factors {2: 1.3, 4: 1.0, 8: 0.85, 16: 0.75}).

Run: `bash configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/r6/run_135.sh`

| # | job | change from the reference | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- |
| 1 | global_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 2 | local_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 3 | hybrid_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 4 | global_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 5 | local_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 6 | hybrid_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 7 | global_ref | none (the reference) | 53 | 0.0-2.2 |
| 8 | local_ref | none (the reference) | 53 | 0.0-2.2 |
| 9 | hybrid_ref | none (the reference) | 53 | 0.0-2.2 |
| 10 | global_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 11 | global_ref_s44 | training_seed 44 (reference 42) | 53 | 0.0-2.2 |
| 12 | local_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 13 | local_ref_s44 | training_seed 44 (reference 42) | 53 | 0.0-2.2 |
| 14 | hybrid_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 15 | hybrid_ref_s44 | training_seed 44 (reference 42) | 53 | 2.2-4.4 |
| 16 | global_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 17 | global_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 2.2-4.4 |
| 18 | global_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 2.2-4.4 |
| 19 | local_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 20 | local_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 2.2-4.4 |
| 21 | local_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 2.2-4.4 |
| 22 | hybrid_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 23 | hybrid_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 4.3-6.5 |
| 24 | hybrid_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 4.3-6.5 |
| 25 | global_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.3-5.5 |
| 26 | local_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.4-5.6 |
| 27 | hybrid_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.4-5.6 |
| 28 | global_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on global_ref | 4 | 4.4-4.6 |
| 29 | global_ref_t0.5 | sampling only: prior_temperature 0.5 on global_ref | 2 | 4.4-4.5 |
| 30 | global_ref_t2 | sampling only: prior_temperature 2 on global_ref | 2 | 4.4-4.5 |
| 31 | local_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on local_ref | 4 | 4.4-4.6 |
| 32 | local_ref_t0.5 | sampling only: prior_temperature 0.5 on local_ref | 2 | 4.4-4.5 |
| 33 | local_ref_t2 | sampling only: prior_temperature 2 on local_ref | 2 | 4.4-4.5 |
| 34 | hybrid_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on hybrid_ref | 4 | 4.5-4.7 |
| 35 | hybrid_ref_t0.5 | sampling only: prior_temperature 0.5 on hybrid_ref | 2 | 4.5-4.6 |
| 36 | hybrid_ref_t2 | sampling only: prior_temperature 2 on hybrid_ref | 2 | 4.5-4.6 |
| 37 | global_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 4.5-7.3 |
| 38 | global_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 4.6-6.5 |
| 39 | global_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 4.6-6.3 |
| 40 | local_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 4.6-7.4 |
| 41 | local_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 4.6-6.5 |
| 42 | local_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 4.7-6.3 |
| 43 | hybrid_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 5.5-8.3 |
| 44 | hybrid_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 5.6-7.5 |
| 45 | hybrid_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 5.6-7.2 |
| 46 | global_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.1-8.3 |
| 47 | global_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.1-8.3 |
| 48 | local_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.1-8.3 |
| 49 | local_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.3-8.5 |
| 50 | hybrid_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.3-8.6 |
| 51 | hybrid_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.5-8.7 |
| 52 | global_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.3 |
| 53 | local_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.4 |
| 54 | hybrid_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.4 |
| 55 | global_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.2-9.4 |
| 56 | global_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 7.3-9.5 |
| 57 | global_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 7.3-9.6 |
| 58 | local_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.4-9.6 |
| 59 | local_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 7.4-9.6 |
| 60 | local_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 7.4-9.6 |
| 61 | hybrid_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.5-9.7 |
| 62 | hybrid_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 8.3-10.5 |
| 63 | hybrid_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 8.3-10.5 |
| 64 | global_ld8 | global latent vector size: vae_latent_dim 8 (reference 32) | 53 | 8.3-10.5 |
| 65 | global_ld16 | global latent vector size: vae_latent_dim 16 (reference 32) | 53 | 8.3-10.5 |
| 66 | global_ld64 | global latent vector size: vae_latent_dim 64 (reference 32) | 53 | 8.5-10.7 |
| 67 | global_ld128 | global latent vector size: vae_latent_dim 128 (reference 32) | 53 | 8.6-10.8 |
| 68 | hybrid_ld16 | global latent vector size: vae_latent_dim 16 (reference 32) | 53 | 8.7-10.9 |
| 69 | hybrid_ld64 | global latent vector size: vae_latent_dim 64 (reference 32) | 53 | 9.4-11.7 |
| 70 | local_ch1 | latent channels per coarse node: latent_ch 1 (reference 4) | 53 | 9.5-11.7 |
| 71 | local_ch2 | latent channels per coarse node: latent_ch 2 (reference 4) | 53 | 9.6-11.8 |
| 72 | local_ch8 | latent channels per coarse node: latent_ch 8 (reference 4) | 53 | 9.6-11.8 |
| 73 | local_ch16 | latent channels per coarse node: latent_ch 16 (reference 4) | 53 | 9.6-11.8 |
| 74 | hybrid_ch2 | latent channels per coarse node: latent_ch 2 (reference 4) | 53 | 9.6-11.8 |
| 75 | hybrid_ch8 | latent channels per coarse node: latent_ch 8 (reference 4) | 53 | 9.7-11.9 |
| 76 | global_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.5-12.7 |
| 77 | global_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.5-12.7 |
| 78 | local_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.5-12.7 |
| 79 | local_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.5-12.7 |
| 80 | hybrid_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.7-12.9 |
| 81 | hybrid_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.8-13.0 |
| 82 | global_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 10.9-13.1 |
| 83 | global_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.7-13.9 |
| 84 | local_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 11.7-13.9 |
| 85 | local_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.8-14.0 |
| 86 | hybrid_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 11.8-14.0 |
| 87 | hybrid_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.8-14.0 |
| 88 | local_fb2 | coarse FM prior blocks: prior_fm_blocks 2 (reference 4) | 53 | 11.8-14.1 |
| 89 | local_fb8 | coarse FM prior blocks: prior_fm_blocks 8 (reference 4) | 53 | 11.9-14.1 |
| 90 | hybrid_fb2 | coarse FM prior blocks: prior_fm_blocks 2 (reference 4) | 53 | 12.7-14.9 |
| 91 | hybrid_fb8 | coarse FM prior blocks: prior_fm_blocks 8 (reference 4) | 53 | 12.7-14.9 |
| 92 | global_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 12.7-14.9 |
| 93 | global_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 12.7-15.0 |
| 94 | local_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 12.9-15.1 |
| 95 | local_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 13.0-15.2 |
| 96 | hybrid_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 13.1-15.3 |
| 97 | hybrid_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 13.9-16.1 |
| 98 | ref_det | deterministic HI-MGN (use_vae False) on the same backbone: the line a sampler must beat | 33 | 13.9-15.3 |
