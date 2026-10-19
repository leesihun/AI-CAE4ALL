**고정된 관측 입력에서 발생하는 산포의 one-to-many 예측: AI 및 통계모델 선례 조사**

조사 기준일: 2026-10-12 (Asia/Seoul).

사용자의 최종 정의를 적용했다. 관측 입력 x를 고정하고, random seed·미관측 미세상태·실제 반복시편만 바꿨을 때 출력 y가 달라지는 문제를 대상으로 한다. 찾는 모형은 평균 f(x)만이 아니라 conditional law p(y|x)를 예측하고 y₁,…,yₘ을 생성하거나 표본 추출할 수 있는 모형이다. time-step evolution을 주목적으로 하는 난류/기상/분자 dynamics 논문은 핵심 목록에서 줄였다.

AI generator가 samples를 생성하는 데 사용하는 noise는 실제 원인의 관측값일 필요가 없다. 반면 실제 imperfection/미세조직/forcing realization 자체를 모두 입력으로 주어 deterministic response를 예측하는 논문은 이 정의와 다르다. 불확실한 물리 입력을 별도로 sampling한 다음 deterministic surrogate에 넣는 방법과, 그 숨겨진 입력을 marginalize한 output law를 직접 학습하는 방법도 구분했다.

**주요 발견**

- 고정 nominal design→반복 제작 형상은 Gen-Fab/GUST, 고정 material/loading→response scatter는 VDMN/generative hyperelasticity/PPgNN·PgMDN, 고정 simulator inputs→hidden-seed output distribution은 MDN/GLD/SPCE/spectral emulator가 가장 직접적이다.
- 분포 검증이 특히 구체적인 수치 선례는 MDN wind loads(같은 x에서 300 reference seeds 대 5000 model draws), replicated GLD wind loads(500 reference seeds/test condition), static borehole spectral/MF-GLaM, stochastic Darcy DLL이다.
- 세 요구조건을 모든 대상에 대한 완전한 분포 일치로 인증한 것은 아니다. Gen-Fab의 OOD 분포 지표는 pixel-intensity 요약, GUST는 coverage 부족, VDMN은 nonlinear spread 오차가 크며, 많은 통계모델은 same-domain 검증이다.
- 강한 support-OOD conditional-PDF 비교는 Han/Bahmani MDN의 analytic stationary-law benchmark에서 확인됐다. 산업 실측 예제가 같은 수준으로 검증된 것은 아니다. CPRNF는 OOD PDF를 실제로 평가했지만 실패한다.

**판정 기준**

① Realistic sample: scalar면 physical range/validity, field/curve/geometry면 물리적·형상적 구조를 갖는 realization을 뜻한다. 분포에서 수학적으로 표본을 뽑을 수 있다는 사실과 실제 generated sample의 validity를 실증한 것은 따로 적었다.

② Distribution agreement: 같은 x에서 참조 반복 samples와 generated samples의 PDF/CDF, Wasserstein/energy distance, rank/calibration, joint covariance 등으로 검증하는 것이 가장 직접적이다. pooled marginal histogram, 평균·표준편차, 한 coverage 수치, spatial feature histogram, posterior reconstruction만으로 전체 joint conditional law 일치를 주장하지 않는다. 유한 표본·일부 지표에서의 근사 일치는 exact equality의 증명이 아니다.

③ OOD results: 새 parameter range/geometry family/physical regime/experimental source를 별도로 적었다. random heldout case나 input domain 안의 새 값은 자동으로 OOD가 아니다. 같은 family의 geometry holdout, source holdout, physics-based regime transfer는 범위 밖 extrapolation과 구분했다. OOD validation이 존재하지만 실패한 경우도 결과로 기록했다.

본 목록은 application/method 논문 40편, 통계·ML 원형 논문 7편, 유동 대표 3편을 포함한다. 각 항목의 access 표시는 전문을 검토한 경우와 primary abstract/저자 기관 record까지만 확보한 경우를 구분한다. 세 조건을 충족하지 못한 관련 논문도 빠진 조건을 확인할 수 있도록 유지했다. systematic review의 모든 데이터베이스/문헌을 망라했다는 의미는 아니다.

**핵심 application/method 논문 색인**

| ID | 논문 | 연도 | 생성/예측 대상 |
|---|---|---|---|
| 01 | [Gen-Fab: A Variation-Aware Generative Model for Predicting Fabrication Variations in Nanophotonic Devices](https://arxiv.org/html/2603.11505v1) | 2026 | 제작·재료·정적 응답 |
| 02 | [GUST: Quantifying Free-Form Geometric Uncertainty of Metamaterials Using Small Data](https://arxiv.org/html/2506.12051v2) | 2025 preprint / 2026 journal | 제작·재료·정적 응답 |
| 03 | [Microstructure-based variational deep material networks for robust uncertainty quantification in materials digital twins](https://www.nature.com/articles/s41524-026-02054-5) | 2026 | 제작·재료·정적 응답 |
| 04 | [Stochastic deep material networks as efficient surrogates for stochastic homogenisation of non-linear heterogeneous materials](https://orbi.uliege.be/handle/2268/330920) | 2025 | 제작·재료·정적 응답 |
| 05 | [Predicting Stress–Strain Curve with Confidence: Balance Between Data Minimization and Uncertainty Quantification by a Dual Bayesian Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC11860118/) | 2025 | 제작·재료·정적 응답 |
| 06 | [Generative Hyperelasticity with Physics-Informed Probabilistic Diffusion Fields](https://www.tajtac.com/assets/pdf/diffusion.pdf) | 2024 online / 2025 volume | 제작·재료·정적 응답 |
| 07 | [AlloyVAE: A generative model for complex probabilistic field-to-field relationships in alloys](https://arxiv.org/html/2604.02281v1) | 2026 preprint | 제작·재료·정적 응답 |
| 08 | [Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](https://arxiv.org/html/2509.03340v1) | 2025 preprint | 제작·재료·정적 응답 |
| 09 | [Predictive microstructure image generation using denoising diffusion probabilistic models](https://doi.org/10.1016/j.actamat.2023.119406) | 2023 | 제작·재료·정적 응답 |
| 10 | [Physics-informed machine learning enabled virtual experimentation for 3D printed thermoplastic](https://pubs.rsc.org/en/content/articlehtml/2024/mh/d4mh01022a) | 2024 | 제작·재료·정적 응답 |
| 11 | [Sim-to-real diffusion models for microstructure prediction in metal additive manufacturing](https://link.springer.com/article/10.1007/s00366-026-02290-9) | 2026 | 제작·재료·정적 응답 |
| 12 | [Data-efficient continuous conditional denoising diffusion model for microstructure generation](https://arxiv.org/html/2607.10429v1) | 2026 preprint | 제작·재료·정적 응답 |
| 13 | [Controllable data augmentation and application of multiaxial fatigue experiments by fatigue conditional generative adversative network model](https://www.sciencedirect.com/science/article/pii/S014211232500413X) | 2026 volume / DOI 2025 | 실험 피로 수명·열유동 산포 |
| 14 | [Probabilistic physics-guided machine learning for fatigue data analysis](https://www.sciencedirect.com/science/article/pii/S0957417420310113) | 2021 | 실험 피로 수명·열유동 산포 |
| 15 | [Physics-guided mixture density networks for uncertainty quantification](https://www.sciencedirect.com/science/article/pii/S0951832022004422) | 2022 | 실험 피로 수명·열유동 산포 |
| 16 | [Fatigue property prediction of additively manufactured Ti-6Al-4V using probabilistic physics-guided learning](https://doi.org/10.1016/j.addma.2021.101876) | 2021 | 실험 피로 수명·열유동 산포 |
| 17 | [Data-driven approaches for fatigue prediction of Ti–6Al–4V parts fabricated by laser powder bed fusion](https://doi.org/10.1016/j.ijfatigue.2024.108167) | 2024 | 실험 피로 수명·열유동 산포 |
| 18 | [Estimating Fatigue Curves with the Random Fatigue-Limit Model](https://www.stat.cmu.edu/technometrics/90-00/vol-41-04/v4104277.pdf) | 1999 | 실험 피로 수명·열유동 산포 |
| 19 | [Predicting critical heat flux with uncertainty quantification and domain generalization using conditional variational autoencoders and deep neural networks](https://arxiv.org/html/2409.05790v1) | 2025 | 실험 피로 수명·열유동 산포 |
| 20 | [Development of physics-consistent conditional diffusion model to overcome data scarcity in critical heat flux](https://arxiv.org/html/2511.16207v1) | 2026 | 실험 피로 수명·열유동 산포 |
| 21 | [Data efficiency assessment of generative adversarial networks in energy applications](https://doi.org/10.1016/j.egyai.2025.100501) | 2025 | 실험 피로 수명·열유동 산포 |
| 22 | [Probabilistic Machine Learning Framework for Predicting Critical Heat Flux in Uniformly Heated Vertical Tubes with Uncertainty Quantification](https://scholarsmine.mst.edu/nuclear_facwork/647/) | 2026 | 실험 피로 수명·열유동 산포 |
| 23 | [A physics-constrained framework for fatigue data augmentation of post-weld treated joints: censored S–N estimation with an exploratory generative component](https://link.springer.com/article/10.1007/s10845-026-02975-4) | 2026 | 실험 피로 수명·열유동 산포 |
| 24 | [Probabilistic surrogate modeling of damage equivalent loads on onshore and offshore wind turbines using mixture density networks](https://wes.copernicus.org/articles/9/1885/2024/index.html) | 2024; corrigendum 2025 | 고정 seed 산포와 stochastic simulators |
| 25 | [Data-driven probabilistic surrogate model for floating wind turbine lifetime damage equivalent load prediction](https://wes.copernicus.org/articles/10/2865/2025/index.html) | 2025 | 고정 seed 산포와 stochastic simulators |
| 26 | [Conditional variational autoencoders for probabilistic wind turbine blade fatigue estimation using SCADA data](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2621) | 2021 | 고정 seed 산포와 stochastic simulators |
| 27 | [Replication-based emulation of the response distribution of stochastic simulators using generalized lambda distributions](https://arxiv.org/pdf/1911.09067) | 2020 | 고정 seed 산포와 stochastic simulators |
| 28 | [Emulation of stochastic simulators using generalized lambda models](https://arxiv.org/abs/2007.00996) | 2021 | 고정 seed 산포와 stochastic simulators |
| 29 | [Stochastic polynomial chaos expansions to emulate stochastic simulators](https://arxiv.org/pdf/2202.03344) | 2023 volume | 고정 seed 산포와 stochastic simulators |
| 30 | [A spectral surrogate model for stochastic simulators computed from trajectory samples](https://arxiv.org/pdf/2207.05653) | 2023 | 고정 seed 산포와 stochastic simulators |
| 31 | [MF-GLaM: A multifidelity stochastic emulator using generalized lambda models](https://arxiv.org/html/2507.10303v1) | 2025 preprint; inspected text updated 2026 | 고정 seed 산포와 stochastic simulators |
| 32 | [Reliability analysis for non-deterministic limit-states using stochastic emulators](https://doi.org/10.1016/j.strusafe.2025.102621) | 2025 | 고정 seed 산포와 stochastic simulators |
| 33 | [Seismic fragility analysis using stochastic polynomial chaos expansions](https://arxiv.org/pdf/2208.07747) | 2023 | 고정 seed 산포와 stochastic simulators |
| 34 | [A Multimodal Conditional Mixture Model With Distribution-Level Physics Priors](https://arxiv.org/html/2602.10451v1) | 2026; published 28 May | 고정 seed 산포와 stochastic simulators |
| 35 | [Conditional Pseudo-Reversible Normalizing Flow for Surrogate Modeling in Quantifying Uncertainty Propagation](https://arxiv.org/html/2404.00502v1) | 2025 journal / 2024 preprint | 고정 seed 산포와 stochastic simulators |
| 36 | [Generative Neural Operators through Diffusion Last Layer](https://arxiv.org/html/2602.04139v2) | 2026 preprint | 고정 seed 산포와 stochastic simulators |
| 37 | [DeepFracture: A Generative Approach for Predicting Brittle Fractures with Neural Discrete Representation Learning](https://arxiv.org/html/2310.13344) | 2025 journal | 제작·재료·정적 응답 |
| 38 | [From Sparse AFM Observations to Probabilistic Macroscale Mechanics: A Generative-Physics Framework for Plant Cell Walls](https://arxiv.org/html/2610.06436v1) | 2026 preprint; October | 제작·재료·정적 응답 |
| 39 | [Efficient, scalable emulation of stochastic simulators: A mixture density network based surrogate modeling framework](https://doi.org/10.1016/j.ress.2025.110806) | 2025 | 고정 seed 산포와 stochastic simulators |
| 40 | [Accelerating Seismic Response Distribution Estimation With Scalable Mixture Density Network Stochastic Surrogate Models](https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.70112) | 2026 | 고정 seed 산포와 stochastic simulators |

**제작·재료·정적 응답**

**01. Gen-Fab: A Variation-Aware Generative Model for Predicting Fabrication Variations in Nanophotonic Devices**
2026; Structural and Multidisciplinary Optimization 69:77. [Primary source](https://arxiv.org/html/2603.11505v1); DOI: 10.1007/s00158-026-04272-3.
- **Model description:** Pix2Pix U-Net bottleneck에 16차원 latent noise를 주입한 conditional GAN. adversarial+L1 손실로 명목 layout에서 제작 후 영상의 조건부 분포를 학습.
- **Sample description:** 입력: 동일 GDS layout. 숨겨진 변동: lithography·etching 및 영상 취득 편차. 출력: 반복 제작된 실리콘 nanophotonic 구조의 SEM형 영상. 실제 220 nm SOI 제작 데이터.
- **분포 검증:** 31 GDS–SEM pair를 회전 증강해 124 pair로 훈련. 다른 분포의 Cross/Square/Target 6종에서 실제 반복 35개씩 비교. variance map, intensity histogram KL/Wasserstein; OOD random-match IoU 88.7%, greedy 89.8%. Table 2 Cross-25 KL=0.2657, W=0.1693.
- **OOD:** 훈련 random Fourier patterns와 다른 primitive 형상. VGG feature 거리로 train/test shift를 별도 확인.
- **판정/한계:** 가장 직접적인 실측 제작 산포 선례. C1 확인, C2는 영상 intensity 요약 분포로 제한, C3 형상 OOD 확인. 전체 공간 joint geometry law나 광학성능 분포의 일치를 입증한 것은 아님.
- **접근 범위:** 저자 전문과 출판 정보 확인

**02. GUST: Quantifying Free-Form Geometric Uncertainty of Metamaterials Using Small Data**
2025 preprint / 2026 journal; Journal of Mechanical Design 148(2):021708. [Primary source](https://arxiv.org/html/2506.12051v2); DOI: 10.1115/1.4069971.
- **Model description:** U-Net/attention/SPADE conditional DDPM. 합성 geometric perturbation pretraining 후 실제 제작 사진으로 fine-tuning. 생성 geometry를 homogenization solver에 전달.
- **Sample description:** 입력: 동일 nominal unit-cell geometry. 출력: boundary deformation/dilation 등이 다른 as-fabricated geometry와 그 형상으로 계산한 elastic tensor. 실제 SLA resin 2 mm extrusion 시편 사진, 64×64 representation.
- **분포 검증:** 사진 처리 후 17 designs×64 cells 유지. 15 designs/960 cells로 fine-tune, 2 unseen designs/128 cells로 평가. 실제 형상에서 계산한 C11/C12/C22/C33 KDE 비교. 합성 검증은 30 designs×3000 variants, density/coverage/Wasserstein.
- **OOD:** test nominal designs는 pretraining 및 fine-tuning에 없음. nearest training geometry와 >25% pixel 차이. 같은 design family 내 형상 holdout이며 실제 제작 도메인으로 적응한 모델.
- **판정/한계:** C1 확인, C2 approximate, C3 제한된 geometry holdout. 실제 물성 시험의 산포가 아니라 실측 형상에서 계산한 물성 산포. 합성 coverage는 중간 수준이며 생성 분포가 좁다고 저자가 명시; 실제 평가의 KDE를 완전 일치로 부를 수 없음.
- **접근 범위:** 저자 전문 확인

**03. Microstructure-based variational deep material networks for robust uncertainty quantification in materials digital twins**
2026; npj Computational Materials 12:261; published 9 May 2026. [Primary source](https://www.nature.com/articles/s41524-026-02054-5); DOI: 10.1038/s41524-026-02054-5.
- **Model description:** VDMN: mechanistic laminate homogenization tree의 orientation/volume-fraction parameters를 확률분포로 학습. elastic stiffness likelihood와 analytic moment propagation으로 훈련; sampling하여 nonlinear constitutive responses 생성.
- **Sample description:** 입력: constituent phase properties 및 동일 loading 조건. 숨겨진 변동: 같은 statistical descriptors를 갖는 다른 microstructure realization. 출력: homogenized stiffness 또는 nonlinear stress–strain response. FFT/DNS 30개 microstructure ensemble과 3D printed two-phase polymer tensile specimens.
- **분포 검증:** Fig.2 고정 phase-property input별 30 reference outputs와 25 VDMN samples/PDF, rank calibration. Fig.3 nonlinear response histograms. Fig.4 12 printed tensile specimens와 두 strain 수준의 stress marginals 비교.
- **OOD:** linear-elastic 훈련에서 nonlinear constitutive law 및 loading으로 전이. 알려진 nonlinear law를 inference 단계에서 교체하는 physics 기반 일반화. Fig.2의 input들은 convex hull 안에 있음.
- **판정/한계:** 정적 material-response 산포의 중요한 선례이나 C2에 큰 한계. nonlinear 1% strain에서 SD MRAE tension 57.9%, shear 33.4%. Supplement S5는 실험 12개를 max-min으로 선택해 reference distribution을 더 uniform하게 편향시켰다고 명시. 동일 CAD의 자연 반복 제작 빈도와 완전한 OOD joint-law 재현을 입증한 사례로 쓰기 어려움.
- **접근 범위:** 본문 및 18쪽 Supplementary Information 확인

**04. Stochastic deep material networks as efficient surrogates for stochastic homogenisation of non-linear heterogeneous materials**
2025; Computer Methods in Applied Mechanics and Engineering 441:117994. [Primary source](https://orbi.uliege.be/handle/2268/330920); DOI: 10.1016/j.cma.2025.117994.
- **Model description:** Interaction-Based DMN에서 phase volume fraction과 topology를 분리하고 reference network topology에 확률 변동을 도입. linear elastic homogenization 훈련을 nonlinear material uncertainty propagation으로 확장.
- **Sample description:** 고정 phase properties/volume fraction/loading에서 random SVE fiber organization 및 clustering으로 달라지는 effective response. 실제 반복 실험이 아닌 numerical homogenization.
- **분포 검증:** 확률적 microstructure response를 DNS와 비교하며 공개 data에는 6 SVE geometry 및 nonlinear stochastic analyses 포함. 세부 PDF distance 값은 이번 접근 범위에서 검증하지 못함.
- **OOD:** arbitrary volume fraction 및 nonlinear response capability를 기술. 명시적 train/test support-OOD 분포 비교는 확인하지 못함.
- **판정/한계:** 직접적인 stochastic material-response surrogate 선례. C1/C2는 공개 primary 설명으로 확인하는 수준, C3 및 정확한 분포 성능은 미확인.
- **접근 범위:** 출판사 본문 excerpt, 저자 기관 repository와 공개 data record 확인; 전문 다운로드 실패

**05. Predicting Stress–Strain Curve with Confidence: Balance Between Data Minimization and Uncertainty Quantification by a Dual Bayesian Model**
2025; Polymers 17:550. [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11860118/); DOI: 10.3390/polym17040550.
- **Model description:** curve-type categorical classifier와 curve-feature normal mean/SD regression을 결합. Bayesian weights의 epistemic uncertainty와 feature-output aleatoric uncertainty를 분리하고 spline로 curve 재구성.
- **Sample description:** 고정 4개 injection molding condition에서 다른 isotactic polypropylene tensile specimen의 curve type, yield/fracture feature 및 stress–strain curve. 27 Taguchi conditions, 각 3–7 실험 반복; 23°C, 50 mm/min.
- **분포 검증:** feature probability calibration, prediction intervals 및 curve-type frequencies. test type 예측 70%/30%, 실험 83%/17%; calibration area 약 0.11 aleatoric/0.18 total.
- **OOD:** classifier 25 conditions train/2 test, regression 26/1; 저자는 extrapolation으로 기술. 광범위한 다른 생산 regime 검증은 아님.
- **판정/한계:** 실측 동일 공정 조건 산포에 매우 가까움. C2는 feature marginals와 type probability. C1 full random realistic curve ensemble 및 joint-curve distribution 검증은 제한, C3는 소수 조건 holdout.
- **접근 범위:** primary 전문 index의 방법 및 검증 내용 확인

**06. Generative Hyperelasticity with Physics-Informed Probabilistic Diffusion Fields**
2024 online / 2025 volume; Engineering with Computers 41:51–69. [Primary source](https://www.tajtac.com/assets/pdf/diffusion.pdf); DOI: 10.1007/s00366-024-01984-2.
- **Model description:** Polyconvex strain-energy NODE의 sample-specific weights를 diffusion으로 생성. correlated latent로 spatial material-property fields도 만들고 FEM에 전달.
- **Sample description:** 동일 prescribed deformation에서 달라지는 material stress–stretch curve 및 strain-energy function. synthetic 10000 material population, 실제 15 mice skin의 off-X/off-Y/equibiaxial tests.
- **분포 검증:** 생성 constitutive curves, stress-at-stretch distributions, energy distance, synthetic parameter joint KDE 비교. 샘플별 polyconvexity를 구조적으로 확보.
- **OOD:** 새 P-shaped specimen 등의 FEM 사례는 uncertainty propagation demonstration. 새로운 material/geometry의 실제 reference distribution과 대조한 OOD 평가 없음.
- **판정/한계:** C1 확인, C2 확인하되 실측 15개로 population/tail 정확도는 제한. C3 없음. condition-dependent shape surrogate라기보다 material-population law와 deformation에 대한 응답 sampling.
- **접근 범위:** 저자 전문 PDF 확인

**07. AlloyVAE: A generative model for complex probabilistic field-to-field relationships in alloys**
2026 preprint; arXiv:2604.02281. [Primary source](https://arxiv.org/html/2604.02281v1).
- **Model description:** composition/SRO smoothing과 encoder self-check를 갖는 conditional VAE. coarse field 조건에서 latent sampling으로 stress fields 생성.
- **Sample description:** 동일 CoCrNi composition 및 short-range-order field에서 숨겨진 atomic arrangement에 의해 다른 정적 residual stress fields. MD를 coarse 4³ blocks로 평균한 데이터.
- **분포 검증:** 동일 input에서 3–5 generated fields. reconstruction MRE 5.4%, R² 약 0.96. 보고된 pooled histogram은 posterior reconstruction과 target 비교.
- **OOD:** 명시적 OOD train/test shift 확인되지 않음.
- **판정/한계:** 문제 자체는 정확한 static one-to-many. C1 확인, C2 prior-generated conditional ensemble 검증 부족, C3 미확인. self-check <4%를 stress-distribution fidelity로 해석하면 안 됨.
- **접근 범위:** 전문 확인

**08. Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems**
2025 preprint; arXiv:2509.03340. [Primary source](https://arxiv.org/html/2509.03340v1).
- **Model description:** equivariant GNN/U-Net flow matching과 symmetry matching으로 symmetry-breaking solution branches의 분포를 생성.
- **Sample description:** 동일 beam parameters 및 compression에서 좌/우 buckling branch와 quasi-static deformation paths. branch를 eigenmode perturbation으로 선택한 numerical data; Allen–Cahn benchmark도 포함.
- **분포 검증:** Fig.1 50 generated paths, Table 1 100 samples의 Wasserstein distance. beam plain FM 43.2에서 symmetry matching 14.8로 개선(논문 단위).
- **OOD:** 명시적 parameter-range/geometry distribution shift 검증 확인 못함.
- **판정/한계:** C1/C2 numerical branch distribution 확인, C3 미확인. branch probability의 reference는 symmetry-equivalent allowable solutions; 실제 manufacturing imperfection이 유도하는 branch 발생빈도를 학습한 것으로 확장하면 안 됨.
- **접근 범위:** 전문 확인

**09. Predictive microstructure image generation using denoising diffusion probabilistic models**
2023; Acta Materialia 261:119406. [Primary source](https://doi.org/10.1016/j.actamat.2023.119406); DOI: 10.1016/j.actamat.2023.119406.
- **Model description:** process embeddings를 multi-resolution attention U-Net에 주입하는 DDPM.
- **Sample description:** 동일 cast-forging process/magnification 조건에서 가능한 AZ80 Mg SEM microstructure images. 11 I-beam components의 27 metallographic samples, 131 image classes.
- **분포 검증:** 512×512 samples, FID 및 DRX grain/phase descriptors. unseen morphology error 6.67±0.85%, seen 6.36±0.42%; physical descriptors는 조건별 적어도 10 images.
- **OOD:** 17 classes(13%) withheld; 한 component의 samples, 다른 component의 한 sample 및 추가 magnification을 포함하는 조합 holdout.
- **판정/한계:** C1 확인, C2는 image/within-image morphology statistics, C3 process-combination/magnification holdout. 동일 조건의 specimen-level engineering-response law를 검증한 것은 아님.
- **접근 범위:** 출판사 전문 index 및 방법/검증 내용 확인

**10. Physics-informed machine learning enabled virtual experimentation for 3D printed thermoplastic**
2024; Materials Horizons 11:6028–6039. [Primary source](https://pubs.rsc.org/en/content/articlehtml/2024/mh/d4mh01022a); DOI: 10.1039/D4MH01022A.
- **Model description:** MLP와 8-component PCA curve representation. formulation/molecular descriptors에 fracture-strain variation parameter e_value를 결합.
- **Sample description:** 동일 ink formulation에서 달라지는 tensile stress–strain response. 62 formulations, 각 2–4 actual curves; 50 formulations/180 curves train, 12/36 test.
- **분포 검증:** e=-2…2로 curve ranges 생성, 실제 curves 및 fracture/toughness metrics 비교. e는 actual fracture strain으로부터 산출된 target-informed variation rank.
- **OOD:** heldout formulations/추가 실험은 있으나 명확한 support-OOD distribution protocol은 확인 못함.
- **판정/한계:** C1 curve variability illustration, C2 독립 conditional sample distribution validation 부족. target-derived e_value와 실제 배포 latent sampling을 구분해야 함.
- **접근 범위:** primary 전문 index의 방법 및 검증 내용 확인

**11. Sim-to-real diffusion models for microstructure prediction in metal additive manufacturing**
2026; Engineering with Computers. [Primary source](https://link.springer.com/article/10.1007/s00366-026-02290-9); DOI: 10.1007/s00366-026-02290-9.
- **Model description:** CFD/phase-field images pretraining 후 실제 EBSD로 fine-tuning 및 distillation하는 conditional DDPM.
- **Sample description:** 고정 laser power/speed에서 다른 316L grain microstructures. simulation 및 실제 AM EBSD images.
- **분포 검증:** 조건별 generated images FID와 grain width/length/area/aspect distributions 비교.
- **OOD:** 새 experimental conditions이지만 해당 conditions가 simulation pretraining에는 포함됨.
- **판정/한계:** C1 확인, C2 morphology statistics, C3 experimental-domain transfer. 모든 훈련 데이터에서 unseen condition인 zero-shot OOD로 해석할 수 없음.
- **접근 범위:** 전문 확인

**12. Data-efficient continuous conditional denoising diffusion model for microstructure generation**
2026 preprint; arXiv:2607.10429. [Primary source](https://arxiv.org/html/2607.10429v1).
- **Model description:** condition-dependent covariance embeddings, vicinal loss 및 classifier-free guidance를 갖는 continuous conditional DDPM.
- **Sample description:** 같은 low-carbon steel Mn composition/cooling schedule에서 초기 grain/CA seed에 따라 다른 ferrite/austenite final maps. 40000 simulated images, 256×256.
- **분포 검증:** 10 unseen Mn values에서 각 100 generated+100 CA reference images. grain size/count, phase fraction/interface/banding KDE 비교.
- **OOD:** Mn .0021–.0220 내부의 새 값; 주로 interpolation.
- **판정/한계:** C1/C2 numerical microstructure-statistics 확인. C3 genuine distribution-shift/extrapolation 검증 없음; 높은 Mn에서 오차 인정.
- **접근 범위:** 전문 확인

**37. DeepFracture: A Generative Approach for Predicting Brittle Fractures with Neural Discrete Representation Learning**
2025 journal; Computer Graphics Forum 44:e70002. [Primary source](https://arxiv.org/html/2310.13344); DOI: 10.1111/cgf.70002.
- **Model description:** discrete latent autodecoder/VQ and random segment code with geometry/impulse conditioning.
- **Sample description:** same object/impact→different final 3D fracture patterns. 4 shapes, each 300 BEM collision simulations. graphical fracture synthesis.
- **분포 검증:** fragment volume CDF vs BEM/Mott, counts/normals across 30 collisions. 서로 다른 impacts pooled comparison.
- **OOD:** unseen collision locations/directions/strength visuals, quantitative OOD law 없음; shape별 separate model.
- **판정/한계:** C1 diverse final shapes, C2 fixed-impact occurrence probability 검증 부족, C3 약함. measured engineering fracture law와 구분.
- **접근 범위:** 전문 확인

**38. From Sparse AFM Observations to Probabilistic Macroscale Mechanics: A Generative-Physics Framework for Plant Cell Walls**
2026 preprint; October; arXiv:2610.06436. [Primary source](https://arxiv.org/html/2610.06436v1).
- **Model description:** Stable Diffusion LoRA-generated AFM-like microstructures→elastic property mapping/homogenization→paired effective-property random field and static FEM.
- **Sample description:** fixed macroscopic specimen geometry/loading에서 Gauss-point material assignment만 다시 뽑아 peak stress 산포. actual plant AFM images로부터 microstructures 생성.
- **분포 검증:** 20 macro realizations의 KDE. training 197 images는 6 raw scans의 patches/flips; macro true response distribution 측정 없음. authors explicitly distribution/heldout independence limitations.
- **OOD:** genuine OOD 없음; three validation image train membership/biological identity unreported.
- **판정/한계:** static random-seed pipeline의 최신 후보이지만 C2 ground-truth macro-law 검증 없음, C3 없음. AI-generated variability의 전파만으로 실제 산포를 맞췄다고 할 수 없음.
- **접근 범위:** 전문 확인

**실험 피로 수명·열유동 산포**

**13. Controllable data augmentation and application of multiaxial fatigue experiments by fatigue conditional generative adversative network model**
2026 volume / DOI 2025; International Journal of Fatigue 202:109216. [Primary source](https://www.sciencedirect.com/science/article/pii/S014211232500413X); DOI: 10.1016/j.ijfatigue.2025.109216.
- **Model description:** strain-path conditional CNN GAN with Wasserstein loss (FatCGAN).
- **Sample description:** 동일 multiaxial strain-controlled loading에서 316L stainless steel stress response 및 fatigue life samples. 실제 fatigue experiments.
- **분포 검증:** abstract는 qualitative/quantitative generated-versus-experimental distribution consistency를 보고. augmented data로 downstream RMSE 개선.
- **OOD:** heldout loading regime 또는 조건별 반복 표본 상세가 접근 가능한 primary 내용에 없음.
- **판정/한계:** 직접적인 experimental one-to-many 후보. C1/C2 주장 있음, C3 및 정확한 conditional PDF protocol 미확인; 완전충족으로 인증할 수 없음.
- **접근 범위:** 출판사 abstract/본문 excerpt 확인; 전문 미확보

**14. Probabilistic physics-guided machine learning for fatigue data analysis**
2021; Expert Systems with Applications 168:114316. [Primary source](https://www.sciencedirect.com/science/article/pii/S0957417420310113); DOI: 10.1016/j.eswa.2020.114316.
- **Model description:** PPgNN: fatigue-life conditional mean/variance에 monotonicity/curvature constraints를 주고 failure/runout likelihood로 학습.
- **Sample description:** 같은 stress/loading/environment conditions에서 다른 specimen fatigue lives. 여러 실제 S–N datasets; mean stress/corrosion/frequency extension.
- **분포 검증:** probabilistic P-S-N fitting, variance and derivative consistency를 실제 data로 검증.
- **OOD:** physics-consistent extrapolation formulation은 있으나 heldout OOD의 empirical conditional PDF 검증 미확인.
- **판정/한계:** scalar life scatter를 직접 다루는 AI/통계 혼합 선례. C1 분포에서 수치 sampling 가능, C2 parametric P-S-N/interval evidence, C3 미확인.
- **접근 범위:** 출판사 primary abstract 및 저자 기관 내용 확인

**15. Physics-guided mixture density networks for uncertainty quantification**
2022; Reliability Engineering & System Safety 228:108823. [Primary source](https://www.sciencedirect.com/science/article/pii/S0951832022004422); DOI: 10.1016/j.ress.2022.108823.
- **Model description:** PgMDN: neural network가 conditional mixture weights/means/scales를 예측. physical statistical/derivative constraints를 dynamic penalties로 부과. censored observations를 likelihood에 포함.
- **Sample description:** fatigue stress/strain 및 부식 등 관측 변수 고정에서 다른 life. bridge frequency 등의 engineering cases도 포함.
- **분포 검증:** single-Gaussian과 다른 skewed/multimodal/truncated law를 표현; fatigue P-S-N distributions 및 probabilistic score validation.
- **OOD:** 명시적인 support-OOD conditional distribution comparison은 확인 못함.
- **판정/한계:** 고정 input의 비정규 산포를 학습하는 직접적인 predecessor. 이론적 mixture expressivity와 실제 full-law equality를 구분할 필요.
- **접근 범위:** primary publisher 내용 및 저자 공개본 내용 확인; 전체 출판사 PDF 미확보

**16. Fatigue property prediction of additively manufactured Ti-6Al-4V using probabilistic physics-guided learning**
2021; Additive Manufacturing 39:101876. [Primary source](https://doi.org/10.1016/j.addma.2021.101876); DOI: 10.1016/j.addma.2021.101876.
- **Model description:** PPgNN: log fatigue life의 conditional Gaussian mean/SD, S–N physics constraints, runouts와 missing process data handling.
- **Sample description:** same stress ratio/amplitude 및 scan speed/power/hatch/layer/heat treatment에서 defect/material variability에 따른 fatigue life. 11 references의 18 experimental datasets.
- **분포 검증:** 6 complete+12 incomplete datasets; leave-complete-dataset-out folds, P-S-N predictions 및 log scores.
- **OOD:** source/process dataset holdout이 있으나 일부 folds는 같은 processing parameters를 공유. 모든 fold를 genuine process-OOD라 할 수 없음.
- **판정/한계:** 실제 제조·수명 산포의 명확한 선례. C2 conditional parametric calibration/interval evidence, large repeated-condition empirical PDF comparison은 없음.
- **접근 범위:** 저자 전문 및 기관 record 확인

**17. Data-driven approaches for fatigue prediction of Ti–6Al–4V parts fabricated by laser powder bed fusion**
2024; International Journal of Fatigue 182:108167. [Primary source](https://doi.org/10.1016/j.ijfatigue.2024.108167); DOI: 10.1016/j.ijfatigue.2024.108167.
- **Model description:** PPgNN 2.0 Gaussian mean/SD with physics constraints, crack-initiation classifier.
- **Sample description:** LPBF Ti-6Al-4V loading/process/morphology에서 fatigue lives. literature training과 independent in-house fatigue tests/CT imaging.
- **분포 검증:** external experimental assessment 및 probabilistic mean/SD output. same-condition empirical density matching 지표는 확인되지 않음.
- **OOD:** 외부 시험기관/제작 데이터 검증 후보; exact support/condition split은 미확인.
- **판정/한계:** 우선 추가 전문 확인할 후보. external test가 있다는 사실만으로 full conditional distribution OOD validation이라 부를 수 없음.
- **접근 범위:** primary author institutional record/abstract 확인; 상세 전문 검증 제한

**18. Estimating Fatigue Curves with the Random Fatigue-Limit Model**
1999; Technometrics 41:277–290. [Primary source](https://www.stat.cmu.edu/technometrics/90-00/vol-41-04/v4104277.pdf); DOI: 10.1080/00401706.1999.10485925.
- **Model description:** specimen-specific random fatigue limit와 life noise를 포함하는 statistical likelihood model. runouts를 censoring으로 처리; conditional CDF/survival law를 도출.
- **Sample description:** 동일 applied stress의 실제 specimens에서 다른 fatigue lives 및 failure하지 않을 확률. laminate panel 예시: 5 stress levels×25 specimens.
- **분포 검증:** conditional CDF, probability plots, empirical distribution/KS 및 residual goodness-of-fit. 분포에서 실제 수명 samples를 수치적으로 추출 가능.
- **OOD:** heldout stress/새 material regime에 대한 독립 OOD 실험은 확인되지 않음.
- **판정/한계:** 생성 이미지가 없는 오래된 통계모델도 user 정의의 scalar one-to-many에 해당. C2 repeat-experiment law 검증, C3 없음.
- **접근 범위:** 전문 확인

**19. Predicting critical heat flux with uncertainty quantification and domain generalization using conditional variational autoencoders and deep neural networks**
2025; Annals of Nuclear Energy 220:111502. [Primary source](https://arxiv.org/html/2409.05790v1); DOI: 10.1016/j.anucene.2025.111502.
- **Model description:** conditional VAE encoder/decoder and DNN ensemble comparator.
- **Sample description:** 동일 7 thermal-hydraulic/geometry variables에서 scalar CHF; fixed condition에서 200 latent draws. NRC real experimental database.
- **분포 검증:** predicted-mean errors, relative SD 및 error histogram. generated CHF conditional distribution을 repeated measurement distribution과 비교한 것이 아님.
- **OOD:** test 1411 cases outside training convex hull, 1047 inside. support-OOD 확인.
- **판정/한계:** C1 random output 생성 및 C3 확인. C2 실제 aleatoric law fidelity는 미입증. 작은 latent spread/정확한 mean은 correct physical scatter의 증거가 아님.
- **접근 범위:** 저자 전문 확인

**20. Development of physics-consistent conditional diffusion model to overcome data scarcity in critical heat flux**
2026; Energy and AI 24:100702. [Primary source](https://arxiv.org/html/2511.16207v1); DOI: 10.1016/j.egyai.2026.100702.
- **Model description:** unconditional tabular diffusion과 conditional CHF diffusion. 조건부 모델은 5 TH inputs에서 CHF를 생성.
- **Sample description:** fixed pressure/mass flux/diameter/heated length/outlet quality→scalar CHF, 500 noise draws. NRC 24579 measurements, 59 studies.
- **분포 검증:** unconditional joint data의 marginal/correlation/ECDF/KS validation은 있음. conditional model은 parity/relative SD/physics consistency 중심.
- **OOD:** shuffled 80/10/10 split; genuine OOD 없음.
- **판정/한계:** C1 확인. unconditional joint-data realism을 fixed-x conditional law 재현으로 대신할 수 없음. C2 conditional validation 부족, C3 없음.
- **접근 범위:** 저자 전문 확인

**21. Data efficiency assessment of generative adversarial networks in energy applications**
2025; Energy and AI 20:100501. [Primary source](https://doi.org/10.1016/j.egyai.2025.100501); DOI: 10.1016/j.egyai.2025.100501.
- **Model description:** conditional GAN / conditional WGAN with FNN latent generators; deterministic FNN baseline.
- **Sample description:** fixed 5 thermal/geometry variables→scalar CHF. 21453 real experimental data points.
- **분포 검증:** random holdout Case 1 KL 0.014 cGAN/0.011 wGAN. Case 2 cross-experiment에서는 mean MAPE/R² 중심.
- **OOD:** 한 실험 source로 훈련, 다른 4 experiments 평가. 일부 OOD 매우 실패: Alekseev-trained cGAN MAPE 271.33%.
- **판정/한계:** C3가 실제 존재하는 중요 negative predecessor. pooled/random-case KL과 OOD mean-error를 합쳐 OOD conditional distribution match라고 하면 안 됨.
- **접근 범위:** primary publisher indexed methods/results 및 author institution 확인

**22. Probabilistic Machine Learning Framework for Predicting Critical Heat Flux in Uniformly Heated Vertical Tubes with Uncertainty Quantification**
2026; Nuclear Technology. [Primary source](https://scholarsmine.mst.edu/nuclear_facwork/647/); DOI: 10.1080/00295450.2026.2629142.
- **Model description:** BNN and Natural Gradient Boosting (NGBoost)로 conditional mean 및 predictive distribution/uncertainty 추정.
- **Sample description:** tube conditions 고정에서 CHF scalar. NRC 24579 real experimental points.
- **분포 검증:** random 80/20 test의 nominal 95% intervals에 BNN 94.7%, NGBoost 93.9% 포함. predictive law의 전체 CDF/shape validation은 접근자료에 없음.
- **OOD:** 별도 unseen dataset을 보고하지만 정확한 split 및 외부 law shift 미확인; NGBoost는 해당 unseen data에서 lookup table보다 못한 예외.
- **판정/한계:** 실험 scatter의 probabilistic regression 선례. coverage 하나로 전체 조건부 분포 일치를 입증한 것은 아님.
- **접근 범위:** primary institutional abstract 확인; full text 미확보

**23. A physics-constrained framework for fatigue data augmentation of post-weld treated joints: censored S–N estimation with an exploratory generative component**
2026; Journal of Intelligent Manufacturing. [Primary source](https://link.springer.com/article/10.1007/s10845-026-02975-4); DOI: 10.1007/s10845-026-02975-4.
- **Model description:** Basquin backbone + CVAE conditional residual model, geometry/stress ratio/treatment/log-stress conditioning, generation-band filtering.
- **Sample description:** 동일 post-weld treatment/loading conditions→fatigue-life scalar. preprocessing 후 1476 experimental specimens.
- **분포 검증:** generated residual scatter 평균 0.83× empirical scatter; 일부 그룹만 KS/W criteria 충족. 23 study leave-one-study-out mean RMSE 0.4529→0.4670.
- **OOD:** external-study holdout downstream regression 검증. OOD generated conditional life law의 직접 비교는 부족.
- **판정/한계:** 명확한 one-to-many 모형이나 분포 fidelity 및 augmentation benefit의 negative result. 실패/제한도 검증 결과로 기록해야 함.
- **접근 범위:** 전문 확인

**고정 seed 산포와 stochastic simulators**

**24. Probabilistic surrogate modeling of damage equivalent loads on onshore and offshore wind turbines using mixture density networks**
2024; corrigendum 2025; Wind Energy Science 9:1885–1904. [Primary source](https://wes.copernicus.org/articles/9/1885/2024/index.html); DOI: 10.5194/wes-9-1885-2024.
- **Model description:** MDN이 Gaussian-mixture weights/means/scales를 입력의 함수로 예측. 각 load channel을 별도 모델링, ordinary GPR 비교.
- **Sample description:** 같은 mean wind speed/TI/shear 및 offshore sea statistics→scalar 10-minute DEL 또는 load SD. TurbSim/파랑 seed는 AI input에서 제외. OpenFAST reference.
- **분포 검증:** TEST1 50 conditions×300 reference seeds, TEST2 24×300 seeds. model 5000 draws/condition. conditional histogram, normalized W2, mean/SD/5%/95% quantiles. 동일 x에서 다봉성도 비교.
- **OOD:** TEST1 bounds는 training과 동일; TEST2도 같은 domain. 별도 onshore/offshore models를 transfer-OOD로 셀 수 없음.
- **판정/한계:** user가 말한 random-seed one-to-many에 가장 정확한 AI 선례 중 하나. 모델이 시간 evolution을 forecasting하지 않고 scalar law를 직접 출력. C1/C2 strong numerical evidence, C3 없음. corrigendum은 DEL 계산식/주파수 설명 수정.
- **접근 범위:** 전문 및 2025-06-10 corrigendum 확인

**25. Data-driven probabilistic surrogate model for floating wind turbine lifetime damage equivalent load prediction**
2025; Wind Energy Science 10:2865–2888. [Primary source](https://wes.copernicus.org/articles/10/2865/2025/index.html); DOI: 10.5194/wes-10-2865-2025.
- **Model description:** 4-component Gaussian MDN, seven metocean inputs→conditional DEL distribution.
- **Sample description:** same wind/wave statistics→6 MW floating wind turbine scalar loads; hidden turbulence/wave realization. BHawC–OrcaFlex numerical reference.
- **분포 검증:** 44 reference seed repetitions vs 5000 model draws at fixed conditions; PDFs/W2. blade-root/tower-top은 대체로 좋음, tower-bottom variance overestimated.
- **OOD:** 4 sites의 lifetime assessment는 downstream deployment; external reference conditional distribution OOD validation 없음.
- **판정/한계:** C1/C2 strong but imperfect. temporal field model이 아닌 fixed-input engineering QoI law. C3 없음.
- **접근 범위:** 전문 확인

**26. Conditional variational autoencoders for probabilistic wind turbine blade fatigue estimation using SCADA data**
2021; Wind Energy 24:1122–1139. [Primary source](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2621); DOI: 10.1002/we.2621.
- **Model description:** 15D latent CVAE from three coarse wind statistics to spatial distribution of blade-root cross-section DELs.
- **Sample description:** same SCADA/coarse wind statistics→273 cross-section load outputs; unresolved inflow realization. 2000 LiDAR-informed TurbSim/OpenFAST/BeamDyn simulations.
- **분포 검증:** generated load variation와 validation cases의 spatial scatter 비교. 고정 x large repeated-reference PDF 비교는 제한.
- **OOD:** random 80/20 및 sparse-data evaluation; explicit distribution-shift split 없음.
- **판정/한계:** multi-output engineering sample generator 선례. C1 확인, C2 약함, C3 없음.
- **접근 범위:** primary full-text 내용 확인

**27. Replication-based emulation of the response distribution of stochastic simulators using generalized lambda distributions**
2020; International Journal for Uncertainty Quantification 10:249–275. [Primary source](https://arxiv.org/pdf/1911.09067); DOI: 10.1615/Int.J.UncertaintyQuantification.2020033029.
- **Model description:** 4 GLD parameters 각각 sparse PCE로 회귀. Uniform latent를 learned quantile function에 넣어 fixed-x output sampling.
- **Sample description:** fixed wind speed/TI/shear/density/inclination→5 MW turbine maximum blade-root flapwise bending moment. hidden TurbSim seeds로 서로 다른 출력.
- **분포 검증:** 485 training conditions×50 seeds, 120 test×500 seeds. Fig.19 fixed-x PDFs/histograms, Fig.17–18 quantiles/mean/SD.
- **OOD:** test conditions sampled by same scheme as train.
- **판정/한계:** C1/C2 strong scalar statistical predecessor, C3 없음. 시간 궤적을 생성하지 않고 최종 scalar conditional law를 학습.
- **접근 범위:** 전문 확인

**28. Emulation of stochastic simulators using generalized lambda models**
2021; SIAM/ASA Journal on Uncertainty Quantification 9:1345–1380. [Primary source](https://arxiv.org/abs/2007.00996); DOI: 10.1137/20M1337302.
- **Model description:** GLD parameters를 PCE로 모델링하고 maximum conditional likelihood/feasible GLS를 결합. 각 x에서 replicated training 필요 없음.
- **Sample description:** 같은 observed simulator input에서 hidden random variables에 따라 달라지는 scalar law. 주요 benchmarks는 mathematical finance/epidemiology.
- **분포 검증:** conditional response distributions, mean/variance 및 normalized Wasserstein 비교.
- **OOD:** genuine external-support OOD distribution evaluation 확인 못함.
- **판정/한계:** engineering application 선례보다 핵심 statistical-method predecessor. one observation/input인 훈련도 conditional law learning이 가능함을 보여줌; 반복 validation과 반복 training을 구분.
- **접근 범위:** primary abstract/journal info와 저자 전문 내용 확인

**29. Stochastic polynomial chaos expansions to emulate stochastic simulators**
2023 volume; International Journal for Uncertainty Quantification 13(2):31–52. [Primary source](https://arxiv.org/pdf/2202.03344); DOI: 10.1615/Int.J.UncertaintyQuantification.2022042912.
- **Model description:** PCE in observed X and artificial latent Z plus Gaussian smoothing noise. distributional likelihood로 coefficients 학습.
- **Sample description:** same x→random scalar, including bimodal benchmark. GBM/epidemiology는 방법론 사례이며 engineering specimens 아님.
- **분포 검증:** reference/generated PDFs and normalized W2; bimodal law를 표현.
- **OOD:** 1000 LHS validation points는 같은 input distribution.
- **판정/한계:** C1/C2 methodological evidence, C3 없음. 통상 deterministic PCE와 달리 unobserved randomness를 latent로 생성.
- **접근 범위:** 전문 확인

**30. A spectral surrogate model for stochastic simulators computed from trajectory samples**
2023; Computer Methods in Applied Mechanics and Engineering. [Primary source](https://arxiv.org/pdf/2207.05653); DOI: 10.1016/j.cma.2022.115875.
- **Model description:** fixed-seed response functions를 sparse PCE로 표현한 후 KLE로 압축하고 coefficient law를 Gaussian/KDE/vine copula 등으로 학습.
- **Sample description:** 정적인 borehole water flow. observed rw/hu/kw 고정, hidden R/Tu/Tl/Hl/L 산포→different scalar flow rate.
- **분포 검증:** fixed-input PDF, normalized Wasserstein, covariance 검증.
- **OOD:** genuine external-support OOD experiment 확인 못함.
- **판정/한계:** C1/C2 strong static statistical example, C3 없음. 제목 trajectory는 parameter-space function f_seed(x)의 추적이며 physical time evolution이 아님.
- **접근 범위:** 전문 확인

**31. MF-GLaM: A multifidelity stochastic emulator using generalized lambda models**
2025 preprint; inspected text updated 2026; arXiv:2507.10303. [Primary source](https://arxiv.org/html/2507.10303v1).
- **Model description:** GLD-PCE conditional distribution에 low-fidelity information/correction 결합.
- **Sample description:** same observed rw/hu/kw with hidden five borehole/aquifer parameters→static water-flow law. 추가 seismic peak-drift law.
- **분포 검증:** Fig.4 fixed-condition PDFs, Fig.5 W2; borehole validation 1000 conditions×10000 seeds.
- **OOD:** test input law와 training law 동일, genuine shift 없음.
- **판정/한계:** C1/C2 strong numerical scalar evidence, C3 없음. low/high fidelity는 OOD domain validation과 별개.
- **접근 범위:** 전문 확인

**32. Reliability analysis for non-deterministic limit-states using stochastic emulators**
2025; Structural Safety. [Primary source](https://doi.org/10.1016/j.strusafe.2025.102621); DOI: 10.1016/j.strusafe.2025.102621.
- **Model description:** GLaM/SPCE conditional CDFs를 통해 stochastic limit-state failure probabilities 계산.
- **Sample description:** fixed beam load/length/width/height→static midspan deflection, Young’s modulus는 unobserved random variable. wind-turbine loads도 포함.
- **분포 검증:** static beam의 analytic failure-probability comparison; wind Fig.7 fixed-speed reference/generated histograms.
- **OOD:** extrapolation capability 언급은 있으나 분리된 OOD conditional PDF experiment 미확인.
- **판정/한계:** user 정의에 정확한 static random-parameter example. C2 static case는 주로 probability/QoI validation; C3 미확인.
- **접근 범위:** 최종 출판사 내용 및 title이 다른 arXiv preprint 전문 확인

**33. Seismic fragility analysis using stochastic polynomial chaos expansions**
2023; Probabilistic Engineering Mechanics 72:103413. [Primary source](https://arxiv.org/pdf/2208.07747); DOI: 10.1016/j.probengmech.2023.103413.
- **Model description:** SPCE for log(EDP) conditional on ground-motion descriptors; directly samples law and computes fragility.
- **Sample description:** same seismic intensity/duration/frequency descriptors→different maximum structural response or drift due hidden earthquake realization/seed. 아닌 next-step trajectory prediction.
- **분포 검증:** Fig.1 four fixed inputs' PDFs, normalized W2; validation 400/200 conditions×250 reference replications.
- **OOD:** same input distribution validation; explicit OOD 없음.
- **판정/한계:** C1/C2 strong scalar engineering one-to-many, C3 없음. data generator가 transient simulation이어도 surrogate target은 scalar conditional law.
- **접근 범위:** 전문 확인

**34. A Multimodal Conditional Mixture Model With Distribution-Level Physics Priors**
2026; published 28 May; International Journal for Numerical Methods in Engineering. [Primary source](https://arxiv.org/html/2602.10451v1); DOI: 10.1002/nme.70346.
- **Model description:** conditional MDN mixture representation with component-weighted physical constraints. preprint Gaussian mixtures, final abstract Gaussian/Student-t; CFM comparator.
- **Sample description:** fixed control λ with bounded unobserved imperfection→bifurcated steady-state response. 별도로 fixed slow/coarse variable u1→fast variable u2의 stationary conditional law. real shock Hugoniot data도 모델링.
- **분포 검증:** Fig.3 fixed λ=-0.8/0/0.8 truth-vs-MDN-vs-CFM density. Fig.4 fixed u1=5 및 training 밖 u1=12에서 1000 generated samples와 analytic stationary conditional PDF 대조.
- **OOD:** u1=12 outside training range에 conditional PDF verification. engineering shock example의 OOD empirical law까지 검증한 것은 아님.
- **판정/한계:** non-time-forecast conditional distribution + OOD의 좋은 mathematical benchmark. C1/C2/C3 numerical qualitative evidence 있으나 industrial repeated-measurement validation이나 전체 field joint-law proof는 아님. branch reference sampling rule과 실제 발생빈도 구분.
- **접근 범위:** 저자 preprint 전문 + final publisher abstract 확인; final은 Student-t components도 추가

**35. Conditional Pseudo-Reversible Normalizing Flow for Surrogate Modeling in Quantifying Uncertainty Propagation**
2025 journal / 2024 preprint; Journal of Machine Learning for Modeling and Computing 6(4):1–28. [Primary source](https://arxiv.org/html/2404.00502v1).
- **Model description:** pseudo-reversible neural maps로 noisy physical model의 conditional law를 직접 학습. function/noise form 사전 지식 없이 likelihood와 reversibility losses 사용.
- **Sample description:** fixed observed x→different y=f(x)+epsilon(x), Gaussian/Laplace homoscedastic/heteroscedastic data; geological carbon-storage numerical application.
- **분포 검증:** fixed x에서 20000 generated samples로 PDF/KL 검증. trained x∈[0,1], eval x∈(-1,2).
- **OOD:** Fig.3–5 x=-0.8/1.8 outside training domain explicit validation, 그러나 PDF/KL markedly fails.
- **판정/한계:** exact user problem formulation의 일반 surrogate. C2 ID succeeds, C3 tested failure. OOD 결과가 존재한다는 것과 OOD에서도 분포가 맞는 것은 다름.
- **접근 범위:** preprint 전문과 ORNL publication record 확인

**36. Generative Neural Operators through Diffusion Last Layer**
2026 preprint; arXiv:2602.04139. [Primary source](https://arxiv.org/html/2602.04139v2).
- **Model description:** neural operator가 input-dependent low-rank basis를 만들고, frozen encoder coefficients에서 conditional diffusion/flow-matching하여 output function samples 생성.
- **Sample description:** static stochastic Darcy: same permeability field a(x), hidden random source f(x)→different pressure fields. 이 항목은 정적 Darcy 검증을 중심으로 평가했다.
- **분포 검증:** 10000 training inputs with one output/input; 32 validation/test inputs with 64 true outputs/input. Energy Distance/Sliced Wasserstein/mean/spread errors로 conditional laws 비교. Darcy Table 3의 DLL mean/spread NRMSE는 0.355/0.357로, 근사 분포 재현이며 정확한 일치는 아님.
- **OOD:** stochastic cases에서 explicit distribution-shift OOD protocol 없음. longer deterministic rollout은 별도 평가.
- **판정/한계:** 고차원 same-input random-field distribution의 직접적 AI 선례. C1/C2 numerical validation, C3 없음. theory의 low-rank 해석과 empirical success를 구분.
- **접근 범위:** v1/v2 전문 확인

**39. Efficient, scalable emulation of stochastic simulators: A mixture density network based surrogate modeling framework**
2025; Reliability Engineering & System Safety 257:110806. [Primary source](https://doi.org/10.1016/j.ress.2025.110806); DOI: 10.1016/j.ress.2025.110806.
- **Model description:** MDN stochastic surrogate with multitask regularization and Bayesian hyperparameter search. observed x에서 mixture-distribution parameters 추정.
- **Sample description:** same input parameter에서 repeated stochastic simulator runs yield different output by hidden randomness; academic/engineering cases의 scalar distributions.
- **분포 검증:** publisher reports small-data 200 and larger 1600–3200 samples regimes의 simulator distribution emulation. exact fixed-x repeat counts/metrics는 full text 미확인.
- **OOD:** explicit OOD conditional distribution protocol 미확인.
- **판정/한계:** 문제 정의가 user 요청과 직접 일치하는 중요한 general engineering surrogate paper. exact applications/validation을 추측하여 채우지 않음.
- **접근 범위:** primary publisher abstract/method excerpt 확인; 전문 미확보

**40. Accelerating Seismic Response Distribution Estimation With Scalable Mixture Density Network Stochastic Surrogate Models**
2026; Earthquake Engineering & Structural Dynamics 55:721–742. [Primary source](https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.70112); DOI: 10.1002/eqe.70112.
- **Model description:** MDN conditional EDP-density surrogate. non-replicated training and higher-dimensional ground-motion explanatory variables 지원.
- **Sample description:** same seismicity/ground-motion features→hidden acceleration realization에 따른 different engineering-demand scalar. fixed-input distribution estimation, time-step prediction 아님.
- **분포 검증:** primary abstract는 EDP distribution emulation과 scalability를 보고. detailed PDFs/OOD split은 전문에서 추가 확인 필요.
- **OOD:** access 가능한 abstract로 genuine OOD 확인 불가.
- **판정/한계:** direct seed-scatter engineering predecessor 후보. 세 조건 모두 확인된 논문으로 분류하지 않음.
- **접근 범위:** primary publisher abstract/record 확인; 전문 미확보

**범용 통계 및 ML 방법론: 실제 산포 예측에 사용할 수 있는 원형**

아래는 engineering application에서 세 조건을 충족했다는 판정이 아니라 방법론의 역할 비교다. 통계모델도 conditional distribution에서 sampling할 수 있으므로, one-to-many가 diffusion/GAN에만 있는 기능은 아니다.

| 원형 논문 | Model / sampling | 한계 |
|---|---|---|
| [Mixture Density Networks](https://research.aston.ac.uk/en/publications/mixture-density-networks/), Bishop, 1994 | NN가 mixture weights/means/scales를 예측해 p(y\|x)를 만든다. fixed x에서 mixture component와 noise를 뽑아 scalar/vector sample을 생성. 원 논문의 robot inverse kinematics는 이 질의의 forward experimental-scatter 사례와 별개. | 범용 conditional-density 방법론. engineering 적용/realistic output/OOD 성능은 개별 논문에서 별도 입증해야 한다. |
| [Generalized Additive Models for Location, Scale and Shape (GAMLSS)](https://doi.org/10.1111/j.1467-9876.2005.00510.x), Rigby & Stasinopoulos, 2005 | location/scale뿐 아니라 shape parameter도 input의 additive function으로 회귀. 선택한 분포의 inverse CDF에서 fixed-x sampling. scalar skewness/tails를 다룰 수 있다. | 분포족 선택이 중요. 자유로운 high-dimensional physical field law 및 OOD 보장이 자동으로 생기지 않는다. |
| [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a.html), Duan et al., ICML 2020 | boosted base learners가 conditional distribution parameters를 proper score와 natural gradient로 학습. 평균 추정만이 아니라 predictive distribution을 출력하며 sampling 가능. | 기본 Gaussian이면 단봉성에 제한. 본 조사 CHF 적용은 coverage/mean validation 중심이며 OOD full-density match는 미입증. |
| [Practical heteroskedastic Gaussian process modeling for large simulation experiments](https://arxiv.org/pdf/1611.05902), Binois, Gramacy & Ludkovski, 2018 | mean process와 input-dependent noise variance process를 함께 추정하고 반복실험을 효율적으로 이용. ATO manufacturing/inventory simulator에서 고정 정책/재고 input의 random output을 모델링. | Gaussian observation law 아래 mean/variance 및 proper-score 검증. GP posterior uncertainty of mean와 physical aleatoric noise를 따로 유지해야 한다. Random holdout은 OOD가 아니다. |
| [Quantile Regression Forests](https://www.jmlr.org/papers/v7/meinshausen06a.html), Meinshausen, JMLR 2006 | forest weights로 conditional empirical CDF 및 임의 quantiles를 추정. fixed x에서 weighted response 또는 inverse empirical CDF로 sampling. | scalar distribution baseline으로 유용. 여러 output을 각각 QRF로 학습하면 joint dependence가 보장되지 않으며 observed response support 밖 새 물리모드 생성은 제한적. |
| [Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression](https://www.jmlr.org/papers/v23/21-0585.html), Cevid et al., JMLR 2022 | MMD splitting으로 mean 외 distribution heterogeneity를 탐지하고 forest weights가 multivariate conditional law를 정의. response vector를 함께 weighted sampling해 dependence를 유지. | 범용 statistical foundation. 새로운 full-field realizations를 neural generator처럼 합성하는 모델은 아니며 extrapolation/novel modes 검증 필요. |
| [Stochastic Kriging for Simulation Metamodeling](https://pubsonline.informs.org/doi/10.1287/opre.1090.0754), Ankenman, Nelson & Staum, 2010 | simulation의 intrinsic noise와 unknown response-surface의 extrinsic uncertainty를 구분해 mean performance surrogate 및 실험 설계를 다룬다. | 관련 기준점이지만 원형 자체를 full-output conditional-distribution generator로 분류하면 안 된다. mean의 posterior draws는 replicate outcome law와 다르다. |

**유동 대표 사례 세 편**

시간 evolution 모델을 다수 나열하는 대신, 조건에서 상태의 분포를 직접 생성하는 대표만 남겼다.

**[Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks](https://arxiv.org/html/2504.02843v1) — ICLR 2025**

Model: graph/latent diffusion 또는 flow matching; fixed mesh/Re/BC→instantaneous equilibrium velocity/pressure fields.

Validation: Ellipse train Re500–1000, thickness0.5–0.8, AoA0. Re400–500/1000–1100, thickness0.45–0.5/0.8–0.9, AoA10° test. PDFs/whole-state graph-wise Wasserstein-2 및 realistic OOD fields.

Scope: 모델 자체는 time-step evolution이 아니지만 reference randomness는 unobserved long-time flow phase. 실험적 제작/재료 산포와 다른 대표 사례.

**[One Scale at a Time: Scale-Autoregressive Modeling for Fluid Flow Distributions](https://arxiv.org/html/2604.11403v1) — 2026 preprint**

Model: mesh hierarchy coarse-to-fine scale-autoregressive flow matching; same observed geometry/BC→complete instantaneous fluid-state samples.

Validation: 동일 EllipseFlow benchmark HighRe/AoA10 OOD W2 및 sample validation.

Scope: scale autoregression은 physical time evolution이 아니다. 다만 DGN과 같은 benchmark/data 계열이므로 독립 실험 증거 두 건으로 과대계산하지 않는다.

**[From Zero to Turbulence: Generative Modeling for 3D Flow Simulation](https://proceedings.iclr.cc/paper_files/paper/2024/file/15972a9575e0f03bf82f00aebeb40774-Paper-Conference.pdf) — ICLR 2024**

Model: boundary/geometry-conditioned DDPM→3D fully-developed turbulent states.

Validation: 27 training geometries, 9 unseen geometries; condition별 16 generated/16 reference states, W2-TKE와 velocity/vorticity distribution metrics.

Scope: physical experiment가 아니라 CFD long-time states의 분포. latest user scope에 따라 대표 선례로만 유지.

**이 문헌들을 인용할 때의 정확한 구분**

1. measured repeat scatter: Gen-Fab, GUST, polymer stress–strain, fatigue models 등이 실제 제작/시험 데이터로 variability를 다룬다. 그러나 output은 SEM/geometry/curve features/scalar life로 서로 다르다.
2. numerical repeat scatter: GLD/SPCE/MDN/spectral/DLL은 simulator reference law를 학습한다. simulator law를 맞추는 것을 자연계의 실제 실험 law를 맞춘 것으로 바꾸어 말하지 않는다.
3. physical sample validity와 statistical frequency validity는 별개다. EFM의 buckling branches가 admissible하더라도 실제 imperfection distribution이 좌/우 branch를 얼마나 자주 만드는지까지 맞췄다는 뜻은 아니다.
4. Bayesian model weights/MC-dropout의 산포는 주로 model uncertainty다. 별도의 likelihood noise/aleatoric generator가 검증되어야 actual repeat outcome scatter와 연결할 수 있다.
5. target-informed VAE posterior reconstruction은 deployment prior sampling과 다르다. AlloyVAE 등의 reconstruction metric을 conditional generation quality로 대체하지 않는다.
6. several scalar distributions를 맞춰도 vector/field joint distribution이 맞았다는 뜻은 아니다. node-wise Gaussian or feature marginals만으로 realistic spatial correlation이 생성되는지 별도 확인해야 한다.
7. training에서 exact x repetition이 없어도 conditional law는 smoothness/shared structure 아래 학습 가능하다. 그러나 이 조사에서 가장 직접적인 evidence는 heldout x에서 실제/수치 repetitions를 확보해 predicted law와 비교한 경우다.

**읽기 우선순위**

- 실제 제작 산포 및 OOD: 01 Gen-Fab, 02 GUST — 형상/영상 output이며 분포 지표의 한계까지 함께 읽을 것.
- 실제 재료 응답 산포: 03 VDMN, 05 Dual Bayesian, 06 Generative Hyperelasticity, 15 PgMDN, 18 Random Fatigue-Limit — 전체 curve/feature/scalar law의 차이를 확인할 것.
- 정확한 fixed-input hidden-seed definition: 24 wind MDN, 27 replicated GLD, 30 spectral borehole, 31 MF-GLaM, 36 DLL Darcy.
- OOD distribution validation의 성공/실패를 비교: 34 conditional MDN benchmark, 35 CPRNF failure, 19/21 CHF mean-only OOD 및 23 external-study fatigue negative result.

이 범위에서 확인되는 연구 공백은 engineering 전체에서 one-to-many가 처음이라는 것이 아니라, 특정 응용에서 현실적인 고차원 realization, 전체 조건부 분포의 충실한 재현, 충분히 분리된 OOD 조건의 분포 검증을 모두 함께 확보하는 데 있다.
