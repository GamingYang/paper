# Degradation-Aware Physics-Guided Hybrid Transfer Learning for Cross-Batch State-of-Health Prediction of Lithium-Ion Batteries

Jichao Zhuang<sup>1,*</sup>, Jiaming Yang<sup>1</sup>

<sup>1</sup> School of Mechanical and Electrical Engineering, Guangzhou University, Guangzhou 510006, China

<sup>*</sup>Corresponding author: J. Zhuang (zjc_10242021@gzhu.edu.cn)

## Abstract

State-of-health (SOH) prediction models trained on historical lithium-ion battery batches often show reduced performance on a new batch. This reduction can arise from distribution shifts, inconsistent degradation-stage coverage, and cell-to-cell variation. To address these challenges, we propose a degradation-aware physics-guided hybrid transfer learning framework for cross-batch SOH prediction. The framework first uses a physical-residual SOH modeling strategy. A low-order nonlinear degradation prior represents the global capacity-fade trend, while a residual temporal module learns cell-specific local deviations from this trend. A degradation-state-guided curve attention mechanism then incorporates the estimated degradation state and local voltage-curve morphology into temporal feature aggregation. This mechanism adjusts the emphasis on local temporal segments across degradation stages. We further introduce lifecycle-indexed degradation-aware maximum mean discrepancy (MMD) regularization to support feature transfer between the source and target domains. Estimated SOH serves as a degradation-state indicator that adjusts alignment strength and increases the cross-domain constraint at lower SOH levels. Experiments on a lithium-ion battery dataset show that the framework provides competitive prediction errors relative to the representative recent methods evaluated here. Ablation results further identify the distinct contributions and limitations of the retained modules.

**Keywords:** lithium-ion battery; state of health; cross-batch prediction; physics-guided learning; transfer learning

## 1 Introduction

Lithium-ion batteries are widely used in electric vehicles, energy-storage systems, and portable electronics because of their high energy density, long cycle life, and low self-discharge rate. Repeated cycling causes capacity fade and internal-resistance growth, which can affect system safety and reliability. Accurate estimation of battery state of health (SOH) is therefore a central task in battery management [1].

Data-driven SOH estimation methods have advanced in recent years. Representative approaches include multimodel fusion [2], data augmentation [3], graph neural networks [4], and hybrid feature networks [5]. These methods seek to improve predictive accuracy and robustness. However, many rely on broadly consistent data distributions between the training and test domains. When batteries come from different production batches or operating conditions, cross-domain shifts can limit transfer of the learned mapping. The resulting loss of generalization may be substantial.

Transfer learning has become an important strategy for addressing distribution shifts in cross-battery SOH estimation. Existing studies have introduced domain adaptation [6], meta-learning [7], deep transfer learning [8], and transfer ensemble learning [9]. These strategies address limited target-domain labels and cross-domain distribution differences. However, many methods still learn the SOH mapping in a fully data-driven manner. The model must then capture the global trend, local degradation patterns, and cross-domain alignment simultaneously. This complex mapping can limit accuracy when target samples are scarce and degradation trajectories differ appreciably.

Physics-guided modeling has also received increasing attention as a way to improve the reliability of SOH estimation. Prior work has examined physics-informed neural networks [10], physical constraints [11], domain-knowledge fusion [12], and degradation-feature optimization [13]. These methods combine degradation knowledge with data-driven models to improve physical consistency. However, they often emphasize physical priors and local degradation information without directly addressing cross-batch distribution shifts. A unified treatment of the global trend, local residuals, and cross-domain feature alignment remains limited.

This study addresses cross-batch lithium-ion battery SOH prediction within a common data system. We propose a degradation-aware physics-guided hybrid transfer learning framework. Its physical-residual strategy uses a low-order nonlinear prior to represent global capacity fade. A residual temporal module learns cell-specific local deviations around this reference trend. We also develop degradation-state-guided curve attention for temporal aggregation. It conditions the aggregation of voltage-curve morphology on the estimated degradation state. Finally, lifecycle-indexed degradation-aware MMD regularization increases feature-alignment strength as estimated SOH decreases. This term provides an additional domain-adaptation constraint during deeper degradation.

The main contributions are as follows.

1. We introduce a physical-residual SOH modeling strategy. A low-order nonlinear prior represents the global degradation trend, while a residual temporal network learns local cell-specific deviations. This division reduces reliance on an entirely black-box SOH mapping.

2. We construct degradation-state-guided curve attention for temporal aggregation. Estimated degradation state and local voltage-curve morphology jointly inform aggregation, allowing attention to vary across degradation stages. The module serves as an auxiliary representation component for stage-dependent variations and local fluctuations.

3. We design lifecycle-indexed degradation-aware MMD regularization. Estimated SOH adjusts the cross-batch alignment strength, imposing a stronger source-target distribution constraint as degradation progresses. This regularizer provides a bounded domain-adaptation contribution to cross-batch SOH prediction.

## 2 Related Work

### 2.1 Lithium-ion battery SOH estimation

The central objective of lithium-ion battery SOH estimation is to map cycling information to capacity retention. Data-driven approaches increasingly learn degradation patterns from voltage curves, capacity changes, partial charge-discharge segments, and multiscale health features [14–16]. With deep learning, the field has further shifted from manually designed features to end-to-end representation learning [17,18]. Many methods nevertheless assume similar distributions in the source and target domains. Transfer learning has therefore been introduced to relax this assumption. Zhang et al. used MMD-assisted alignment when source and target feature spaces differed, improving one-shot battery health estimation [19]. Given source domain <img src="zjc8.4_SOH_prediction_English_assets/image1.png" alt="source domain" height="22"> and target domain <img src="zjc8.4_SOH_prediction_English_assets/image2.png" alt="target domain" height="22">, transfer learning seeks to adapt source-domain health representations to the target domain, as illustrated in Fig. 1. A general objective is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image3.png" alt="Equation 1" width="520"></p>
<p align="center">(1)</p>

where <img src="zjc8.4_SOH_prediction_English_assets/image4.png" alt="source-domain SOH regression loss" height="22"> is the source-domain SOH regression loss, <img src="zjc8.4_SOH_prediction_English_assets/image5.png" alt="adaptation constraint" height="22"> is the adaptation constraint between domains, and <img src="zjc8.4_SOH_prediction_English_assets/image6.png" alt="balancing coefficient" height="22"> is a balancing coefficient.

Existing cross-batch studies have examined lightweight transfer learning [20], few-shot transfer learning [21], and time-frequency domain adaptation [22]. However, when target samples are limited and degradation-stage coverage is incomplete, the mapping <img src="zjc8.4_SOH_prediction_English_assets/image7.png" alt="mapping function" height="22"> may become unstable during late-stage degradation. Cross-batch SOH prediction therefore requires both a transfer mechanism and a suitable degradation-trend prior that structurally constrains the output.

![Conceptual comparison between source-only and transfer-learning-based SOH estimation](zjc8.4_SOH_prediction_English_assets/image8.png)

**Fig. 1.** Conceptual comparison between source-only SOH estimation and transfer-learning-based SOH estimation.

### 2.2 Physics-constrained SOH estimation

Physical constraints can reduce overfitting under limited data and incorporate physical knowledge, domain priors, or distribution constraints into SOH prediction. As illustrated in Fig. 2, physical embedding can combine observed features with prior-derived features and add a corresponding constraint to the training objective:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image9.png" alt="Equation 2" width="520"></p>
<p align="center">(2)</p>

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image10.png" alt="Equation 3" width="520"></p>
<p align="center">(3)</p>

where <img src="zjc8.4_SOH_prediction_English_assets/image11.png" alt="observed degradation features" height="22"> denotes observed battery-degradation features, <img src="zjc8.4_SOH_prediction_English_assets/image12.png" alt="embedded prior features" height="22"> denotes features constructed from physical knowledge or degradation priors, and <img src="zjc8.4_SOH_prediction_English_assets/image13.png" alt="physical constraint" height="22"> is a physical-consistency or degradation constraint.

Recent studies have incorporated physical knowledge, hybrid modeling, and degradation-feature constraints into SOH estimation [23–26]. These approaches combine empirical degradation regularities with data-driven models to improve accuracy and physical consistency. Feature selection, diagnostic-test design, and generative learning also introduce degradation priors from different perspectives [3,27–29]. However, most studies focus on specific priors or local degradation information and give less attention to cross-batch distribution shifts. A framework that jointly models degradation-trend constraints and cross-domain alignment is still needed. The remaining question is how a physical prior can provide structural guidance during transfer learning.

![Schematic view of regularized learning for battery SOH prediction](zjc8.4_SOH_prediction_English_assets/image14.png)

**Fig. 2.** Schematic view of regularized learning for battery SOH prediction.

## 3 Methods

### 3.1 Problem definition

Consider the <img src="zjc8.4_SOH_prediction_English_assets/image15.png" alt="sample index" height="22">th sample, which contains a single-cycle charging-voltage trajectory and its cycle position. Linear interpolation resamples the voltage trajectory as a one-dimensional sequence <img src="zjc8.4_SOH_prediction_English_assets/image17.png" alt="voltage sequence" height="22"> of length <img src="zjc8.4_SOH_prediction_English_assets/image16.png" alt="sequence length" height="22">. Its cycle position is denoted by <img src="zjc8.4_SOH_prediction_English_assets/image18.png" alt="cycle position" height="22">. Nominal SOH is the prediction target and is defined as <img src="zjc8.4_SOH_prediction_English_assets/image19.png" alt="nominal SOH definition" height="22">. Here, <img src="zjc8.4_SOH_prediction_English_assets/image20.png" alt="discharge capacity" height="22"> is the discharge capacity at cycle <img src="zjc8.4_SOH_prediction_English_assets/image21.png" alt="cycle index" height="22">, and <img src="zjc8.4_SOH_prediction_English_assets/image22.png" alt="rated capacity" height="22"> is the rated capacity. The normalized cycle position is <img src="zjc8.4_SOH_prediction_English_assets/image23.png" alt="normalized cycle position" height="22">, where <img src="zjc8.4_SOH_prediction_English_assets/image24.png" alt="normalization scale" height="22"> is the normalization scale. The model input is therefore <img src="zjc8.4_SOH_prediction_English_assets/image25.png" alt="model input" height="22">. Cross-batch SOH prediction seeks a mapping <img src="zjc8.4_SOH_prediction_English_assets/image26.png" alt="mapping function" height="22"> that is constrained by both source and target domains. The resulting estimate is <img src="zjc8.4_SOH_prediction_English_assets/image27.png" alt="SOH estimate" height="22">, where <img src="zjc8.4_SOH_prediction_English_assets/image28.png" alt="model parameters" height="22"> denotes the model parameters.

### 3.2 Overall framework

We propose a degradation-aware physics-guided hybrid transfer learning framework for cross-batch lithium-ion battery SOH prediction. As shown in Fig. 3, the framework contains three main components. These are physical-residual SOH modeling, degradation-state-guided curve attention, and lifecycle-indexed degradation-aware MMD regularization. First, normalized cycle position defines a low-order nonlinear degradation prior. A residual temporal module then learns local cell-specific deviations from this global trend. Second, curve attention adjusts the emphasis on local sequence segments according to degradation stage. Finally, degradation-aware MMD provides a consistency constraint on cross-batch features.

The predicted SOH is expressed as the sum of a low-order degradation prior and a data-driven residual correction:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image29.png" alt="Equation 4" width="520"></p>
<p align="center">(4)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image30.png" alt="low-order nonlinear degradation prior" height="22"> denotes the low-order nonlinear degradation prior, <img src="zjc8.4_SOH_prediction_English_assets/image31.png" alt="residual term" height="22"> is the residual term, and <img src="zjc8.4_SOH_prediction_English_assets/image32.png" alt="residual-network parameters" height="22"> denotes the residual-network parameters. The training objective combines SOH regression with a feature-alignment regularizer:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image33.png" alt="Equation 5" width="520"></p>
<p align="center">(5)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image34.png" alt="regression loss" height="22"> is the regression loss. <img src="zjc8.4_SOH_prediction_English_assets/image35.png" alt="degradation-aware regularization loss" height="22"> denotes the degradation-aware regularization loss between source domain <img src="zjc8.4_SOH_prediction_English_assets/image36.png" alt="source domain" height="22"> and target domain <img src="zjc8.4_SOH_prediction_English_assets/image37.png" alt="target domain" height="22">. The coefficient <img src="zjc8.4_SOH_prediction_English_assets/image38.png" alt="balancing coefficient" height="22"> balances the two terms.

![Overall framework](zjc8.4_SOH_prediction_English_assets/image39.png)

**Fig. 3.** Overall framework.

### 3.3 Physical-residual SOH modeling

We decompose battery SOH degradation into a shared trend and cell-specific local deviations. This decomposition reduces the burden of fitting the entire SOH curve with a neural network. A low-order nonlinear prior first represents the global capacity-fade trend over cycling. The residual temporal module then compensates for local deviations that the prior cannot describe.

#### 3.3.1 Construction of the low-order nonlinear degradation prior

Lithium-ion battery capacity commonly follows a nonlinear declining trend, although local degradation differs among cells. A neural network fitted directly to SOH must represent the global trend, local fluctuations, and cross-batch differences simultaneously. This requirement increases learning complexity and can weaken physical consistency. We instead formulate a nonlinear degradation prior as a low-order function projection. Normalized cycle position is mapped into a low-dimensional degradation basis to obtain a shared SOH reference baseline.

For sample <img src="zjc8.4_SOH_prediction_English_assets/image40.png" alt="sample index" height="22"> with normalized cycle position <img src="zjc8.4_SOH_prediction_English_assets/image41.png" alt="normalized cycle position" height="22">, the low-order degradation basis vector is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image42.png" alt="Equation 6" width="640"></p>
<p align="center">(6)</p>

The low-order nonlinear degradation prior is the projection of this basis vector onto a coefficient vector:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image43.png" alt="Equation 7" width="520"></p>
<p align="center">(7)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image44.png" alt="coefficient vector" height="22"> contains the trainable coefficients <img src="zjc8.4_SOH_prediction_English_assets/image45.png" alt="coefficient a" height="22">, <img src="zjc8.4_SOH_prediction_English_assets/image46.png" alt="coefficient b" height="22">, <img src="zjc8.4_SOH_prediction_English_assets/image47.png" alt="coefficient c" height="22">, and <img src="zjc8.4_SOH_prediction_English_assets/image48.png" alt="coefficient d" height="22">. Together, they determine the baseline offset, overall decline, and nonlinear curvature. The prior therefore represents the SOH trend with few parameters.

Figure 4 visualizes the pairwise optimization landscapes of <img src="zjc8.4_SOH_prediction_English_assets/image45.png" alt="coefficient a" height="22">, <img src="zjc8.4_SOH_prediction_English_assets/image46.png" alt="coefficient b" height="22">, <img src="zjc8.4_SOH_prediction_English_assets/image47.png" alt="coefficient c" height="22">, and <img src="zjc8.4_SOH_prediction_English_assets/image48.png" alt="coefficient d" height="22">. Each fitted coefficient pair lies within a region of low RMSE increase. The error also changes continuously around the optimum. These observations support the use of the low-order projection as a stable global reference within the evaluated source data. The residual module can then focus on cell-specific local departures from that reference.

![Pairwise optimization landscape of the four coefficients](zjc8.4_SOH_prediction_English_assets/image49.png)

**Fig. 4.** Pairwise optimization landscape of the four coefficients in the low-order nonlinear degradation prior. In each panel, two coefficients vary while the other two remain fixed at their source-domain least-squares optima. Color indicates the RMSE increase relative to the optimal coefficient solution, and the red point marks the fitted pair. Coefficients are optimized using only Batch 1–8 source-domain training samples.

To prevent information from the target test batch from entering prior construction, the coefficients are fitted only to source-domain training samples. Let <img src="zjc8.4_SOH_prediction_English_assets/image50.png" alt="source-domain training set" height="22"> denote the source-domain training set and <img src="zjc8.4_SOH_prediction_English_assets/image51.png" alt="true SOH" height="22"> its true SOH values. Coefficient optimization is written as

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image52.png" alt="Equation 8" width="520"></p>
<p align="center">(8)</p>

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image53.png" alt="Equation 9" width="520"></p>
<p align="center">(9)</p>

Equivalently, this procedure finds the least-squares projection <img src="zjc8.4_SOH_prediction_English_assets/image54.png" alt="projection function" height="22"> of source-domain SOH observations in the low-order basis space. This projection captures the shared component representable by the nonlinear basis. The residual temporal module learns the remaining cell-specific local deviations. The prior therefore acts as a structurally interpretable global anchor rather than a complete physical model.

#### 3.3.2 Residual temporal modeling

Cells can still depart from the global trend because of manufacturing variation, cycling conditions, and batch differences. We therefore construct a residual temporal module to represent local degradation features that the low-order prior cannot capture. For sample <img src="zjc8.4_SOH_prediction_English_assets/image55.png" alt="sample index" height="22">, the residual target is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image56.png" alt="Equation 10" width="520"></p>
<p align="center">(10)</p>

The residual-network output is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image57.png" alt="Equation 11" width="520"></p>
<p align="center">(11)</p>

The SOH prediction is therefore

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image58.png" alt="Equation 12" width="520"></p>
<p align="center">(12)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image59.png" alt="local residual correction" height="22"> is the learned local residual correction. The branch uses one-dimensional convolution to extract local temporal features from the charging-voltage trajectory. A recurrent unit then models the reconstructed sequence and its longer-range temporal dependencies. The operations are

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image60.png" alt="Equation 13" width="520"></p>
<p align="center">(13)</p>

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image61.png" alt="Equation 14" width="520"></p>
<p align="center">(14)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image62.png" alt="one-dimensional convolution" height="22"> denotes one-dimensional convolution, <img src="zjc8.4_SOH_prediction_English_assets/image63.png" alt="local voltage-trajectory features" height="22"> denotes local voltage-trajectory features, and <img src="zjc8.4_SOH_prediction_English_assets/image64.png" alt="recurrent unit" height="22"> is the recurrent unit. Its output sequence is <img src="zjc8.4_SOH_prediction_English_assets/image65.png" alt="output sequence features" height="22">. This residual formulation separates the global trend from local deviations. The network consequently learns cell-specific patterns around the baseline instead of fitting the complete SOH curve directly.

### 3.4 Degradation-state-guided curve attention for temporal aggregation

The residual temporal branch encodes the charging-voltage trajectory through local convolution and a recurrent unit. This process yields a sequence of hidden states. For sample *i*, the sequence is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image66.png" alt="Equation 15" width="520"></p>
<p align="center">(15)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image67.png" alt="hidden state at temporal position t" height="22"> is the hidden state at local temporal position *t*, and *T* is the sequence length. Conventional attention generally uses only <img src="zjc8.4_SOH_prediction_English_assets/image68.png" alt="hidden states" height="22"> to estimate positional importance. It may therefore miss stage-dependent differences in the contribution of local curve segments. We construct degradation-state-guided curve attention to provide additional conditioning during temporal aggregation. A degradation-state vector and local curve-morphology token guide the focus on temporal information, as shown in Fig. 5.

![Flowchart of degradation-state-guided curve attention](zjc8.4_SOH_prediction_English_assets/image69.jpeg)

**Fig. 5.** Flowchart of degradation-state-guided curve attention for temporal aggregation.

First, normalized cycle position and the low-order prior output define the degradation-state vector:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image70.png" alt="Equation 16" width="520"></p>
<p align="center">(16)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image71.png" alt="normalized cycle position" height="22"> is normalized cycle position, and <img src="zjc8.4_SOH_prediction_English_assets/image72.png" alt="global degradation reference" height="22"> is the global degradation reference from the low-order prior. The term <img src="zjc8.4_SOH_prediction_English_assets/image73.png" alt="relative degradation level" height="22"> represents relative degradation. This state vector does not replace voltage-trajectory features. Instead, it conditions the attention calculation with information about the sample's degradation stage.

Local curve changes can be smoothed during hidden-state encoding. We therefore construct a curve-morphology token directly from the voltage trajectory:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image74.png" alt="Equation 17" width="520"></p>
<p align="center">(17)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image75.png" alt="voltage at position t" height="22"> is the voltage at position *t*. The terms <img src="zjc8.4_SOH_prediction_English_assets/image76.png" alt="first-order local difference" height="22"> and <img src="zjc8.4_SOH_prediction_English_assets/image77.png" alt="second-order local difference" height="22"> are its first- and second-order local differences. Equation (17) represents local voltage level, slope variation, and curvature. These descriptors provide morphological information for the subsequent attention allocation.

The degradation-state vector is mapped to a stage-conditioned query:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image78.png" alt="Equation 18" width="520"></p>
<p align="center">(18)</p>

The attention score at position *t* then combines the hidden state, curve-morphology token, and stage-conditioned query:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image79.png" alt="Equation 19" width="520"></p>
<p align="center">(19)</p>

The parameters <img src="zjc8.4_SOH_prediction_English_assets/image80.png" alt="learnable parameter" height="22">, <img src="zjc8.4_SOH_prediction_English_assets/image81.png" alt="learnable parameter" height="22">, and <img src="zjc8.4_SOH_prediction_English_assets/image82.png" alt="learnable parameter" height="22"> are learned during training. Unlike attention based only on hidden states, Eq. (19) uses three sources of information. The term <img src="zjc8.4_SOH_prediction_English_assets/image83.png" alt="temporal degradation representation" height="22"> provides the temporal degradation representation learned by the residual branch. The term <img src="zjc8.4_SOH_prediction_English_assets/image84.png" alt="local curve morphology" height="22"> provides local curve morphology, while <img src="zjc8.4_SOH_prediction_English_assets/image85.png" alt="stage-conditioned guidance" height="22"> supplies stage-conditioned guidance.

We also define a degradation-state-dependent temperature to avoid a fixed attention sharpness across all stages:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image86.png" alt="Equation 20" width="520"></p>
<p align="center">(20)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image87.png" alt="temperature lower bound" height="22"> is the lower temperature bound, and <img src="zjc8.4_SOH_prediction_English_assets/image88.png" alt="learnable parameter" height="22"> is learnable. Equation (20) allows attention concentration to vary with degradation state. The model can therefore shift between broader aggregation and more concentrated local focus across cycle positions and estimated degradation levels.

Finally, temperature-adjusted normalized weights aggregate the temporal hidden states:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image89.png" alt="Equation 21" width="520"></p>
<p align="center">(21)</p>

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image90.png" alt="Equation 22" width="520"></p>
<p align="center">(22)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image91.png" alt="attention weight" height="22"> is the attention weight at position *t*. The aggregated residual temporal representation <img src="zjc8.4_SOH_prediction_English_assets/image92.png" alt="aggregated temporal representation" height="22"> enters the regression layer to obtain the residual correction. This module is an auxiliary representation component within the physical-residual strategy. It enhances local temporal aggregation while retaining a compact overall structure.

![Mechanistic visualization of degradation-state-guided adaptive curve attention](zjc8.4_SOH_prediction_English_assets/image93.png)

**Fig. 6.** Mechanistic visualization of degradation-state-guided adaptive curve attention. (a) Representative voltage-curve tokens and high-attention locations, with orange ticks marking locally emphasized segments. (b) Mean attention distributions at different degradation stages, showing more concentrated peaks during the late stage. (c) Attention temperature and entropy decrease as normalized cycle position increases, consistent with a sharper focus as degradation progresses. (d–f) Response surfaces of attention compatibility to voltage level *V*, slope Δ*V*, and curvature Δ²*V* under fixed late-stage degradation and hidden context. Black points denote late-stage, high-attention curve tokens, and the orange point is their attention-weighted center.

### 3.5 Lifecycle-indexed degradation-aware MMD regularization

Historical source batches and a new target batch generally exhibit a feature-distribution shift. Their global degradation patterns can be similar, while voltage trajectories, capacity-fade rates, and stage-specific features still differ. These differences can affect cross-domain generalization. We therefore regularize the latent source and target feature distributions during training. The lifecycle-indexed degradation-aware maximum mean discrepancy (MMD) term constrains cross-batch feature differences, as illustrated in Fig. 7.

![Mechanism of degradation-aware MMD regularization](zjc8.4_SOH_prediction_English_assets/image94.png)

**Fig. 7.** Mechanism of degradation-aware MMD regularization for cross-batch feature alignment.

Given fused source features <img src="zjc8.4_SOH_prediction_English_assets/image95.png" alt="source fused features" height="22"> and fused target features <img src="zjc8.4_SOH_prediction_English_assets/image96.png" alt="target fused features" height="22">, MMD measures their mean embeddings in a reproducing kernel Hilbert space:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image97.png" alt="Equation 23" width="520"></p>
<p align="center">(23)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image98.png" alt="feature mapping" height="22"> is the feature mapping, and <img src="zjc8.4_SOH_prediction_English_assets/image99.png" alt="reproducing kernel Hilbert space" height="22"> is the reproducing kernel Hilbert space. Expanding the kernel yields

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image100.png" alt="Equation 24" width="620"></p>
<p align="center">(24)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image101.png" alt="kernel function" height="22"> is the kernel function. We use a multikernel Gaussian kernel to represent distribution differences at multiple scales:

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image102.png" alt="Equation 25" width="520"></p>
<p align="center">(25)</p>

Conventional MMD applies the same alignment strength across degradation stages. It therefore does not represent stage-dependent cross-batch shifts. Because lower-SOH operation can exhibit greater degradation uncertainty and batch variation, we define an exponential degradation-aware weight. This weight increases alignment strength as the target-domain SOH estimate decreases. For each training batch, mean target-domain SOH is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image103.png" alt="Equation 26" width="520"></p>
<p align="center">(26)</p>

The corresponding exponential degradation-aware weight is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image104.png" alt="Equation 27" width="520"></p>
<p align="center">(27)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image105.png" alt="degradation-aware coefficient" height="22"> is the degradation-aware coefficient. The final degradation-aware MMD regularizer is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image106.png" alt="Equation 28" width="520"></p>
<p align="center">(28)</p>

The complete training loss is

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image107.png" alt="Equation 29" width="520"></p>
<p align="center">(29)</p>

The first term is the supervised source-domain SOH loss. The second is the degradation-aware feature-alignment regularizer between the source and target domains.

![Training trajectories of static and degradation-aware weighted MMD losses](zjc8.4_SOH_prediction_English_assets/image108.png)

**Fig. 8.** Training trajectories of static and degradation-aware weighted MMD losses. The gray dashed curve is the static MMD loss without degradation-dependent weighting. The blue curve is the exponential degradation-aware weighted MMD loss. Shading represents mean ± s.e.m. over three random seeds. The highlighted interval is the MMD-guided training stage, and the inset enlarges the late active window.

Figure 8 compares the training trajectories of static MMD and exponential degradation-aware weighted MMD. Both losses decrease during the MMD-guided interval. The weighted loss remains lower in the later, higher-degradation portion of the active interval. The exponential weight does not change the underlying MMD measure. Instead, it adjusts cross-batch alignment strength during training and imposes a stronger constraint at greater degradation levels.

### 3.6 Implementation details

The model takes a 100-point charging-voltage trajectory and normalized cycle position as inputs. Coefficients of the low-order prior are fitted only to source-domain training samples. The first and second convolutional layers map the input to 16 and 32 channels, respectively. Each layer uses one-dimensional convolution with kernel size 3, batch normalization, ReLU activation, and max pooling. The recurrent unit has a hidden dimension of 64. The attention module receives hidden states, the degradation-state vector, and curve-morphology tokens. Each token contains voltage, its first difference, and its second difference. The state vector contains normalized cycle position, the prior output, and relative degradation. Training uses Adam with an initial learning rate of <img src="zjc8.4_SOH_prediction_English_assets/image109.png" alt="initial learning rate" height="22">, a batch size of 128, and 150 epochs. Table 1 lists the remaining hyperparameters. The outer MMD weight is <img src="zjc8.4_SOH_prediction_English_assets/image110.png" alt="outer MMD weight" height="22">, and the degradation-aware coefficient is <img src="zjc8.4_SOH_prediction_English_assets/image111.png" alt="degradation-aware coefficient" height="22">. Algorithm 1 summarizes the training procedure.

**Table 1. Implementation settings and hyperparameters**

| Category | Parameter | Setting | Description |
|---|---|---|---|
| Input | Voltage sequence | 100 points | Interpolated charging-voltage trajectory |
| Input | Cycle position | Normalized *p* | Used for degradation prior and state vector |
| Physical prior | Prior form | Cubic polynomial | Fitted using source-domain samples |
| CNN block | Convolutional channels | 16 → 32 | Local voltage-feature extraction |
| CNN block | Kernel/pooling | *k* = 3; MaxPool(2) | Applied in each convolutional block |
| Temporal encoder | Recurrent unit | Hidden size = 64 | Encodes local temporal states |
| Attention | State vector | dim = 3 | [*p*, *y*<sub>base</sub>, 1 − *y*<sub>base</sub>] |
| Attention | Curve token | dim = 3 | [*V*, Δ*V*, Δ²*V*] |
| Regression head | Dropout | 0.2 | Used before residual SOH regression |
| Optimizer | Optimizer | Adam | Applied to all trainable modules |
| Training | Epochs/batch size | 150/128 | Fixed for all retained experiments |
| Learning rate | Initial/decay | 1e−3; ×0.1 at epoch 100 | Late-stage training stabilization |
| MMD regularization | Active epochs | 21–120 | Warm-up and sprint stages exclude MMD |
| MMD regularization | λ<sub>MMD</sub>/γ | 5e−5/5.5 | Outer MMD weight and degradation factor |

![Algorithm 1, overall training procedure](zjc8.4_SOH_prediction_English_assets/image112.png)

**Algorithm 1.** Overall training procedure.

## 4 Results

### 4.1 Dataset description

We used the public lithium-ion battery dataset released by the Huazhong Artificial Intelligence Lab. It contains 77 commercial cells and more than 140,000 charge-discharge cycles collected under different discharge protocols. Data were partitioned by battery batch. Batch 1–8 formed the source domain, with 61 cells and 115,256 samples. Batch 9 served as the target-development domain, with eight cells and 16,310 samples. It was used for adaptation and model selection. Batch 10 was reserved as the final blind-test set, with eight cells and 14,556 samples. It did not participate in training or checkpoint selection. Table 2 summarizes this fixed split, and Fig. 9 visualizes the dataset.

Average RMSE, MAE, and MAPE were used to quantify predictive error. These sample-level metrics can be influenced by cells with longer cycling records. We therefore also report Macro RMSE, which averages cell-level prediction errors. Suppose Batch 10 contains <img src="zjc8.4_SOH_prediction_English_assets/image113.png" alt="number of cells" height="22"> cells, and cell <img src="zjc8.4_SOH_prediction_English_assets/image114.png" alt="cell index" height="22"> contains <img src="zjc8.4_SOH_prediction_English_assets/image115.png" alt="number of cycle samples" height="22"> cycle samples. Macro RMSE is defined as

<p align="center"><img src="zjc8.4_SOH_prediction_English_assets/image116.png" alt="Equation 30" width="620"></p>
<p align="center">(30)</p>

Here, <img src="zjc8.4_SOH_prediction_English_assets/image117.png" alt="true SOH" height="22"> and <img src="zjc8.4_SOH_prediction_English_assets/image118.png" alt="predicted SOH" height="22"> are the true and predicted SOH values for cycle <img src="zjc8.4_SOH_prediction_English_assets/image120.png" alt="cycle index" height="22"> of cell <img src="zjc8.4_SOH_prediction_English_assets/image119.png" alt="cell index" height="22">, respectively.

![Overview of the dataset and representative voltage-current degradation signals](zjc8.4_SOH_prediction_English_assets/image121.png)

**Fig. 9.** Overview of the lithium-ion battery dataset and representative voltage-current degradation signals.

**Table 2. Fixed cross-batch split summary**

| Split | Samples | Cells | Batches | Input shape | *y*<sub>min</sub> | *y*<sub>max</sub> |
|---|---:|---:|---|---|---:|---:|
| Source | 115256 | 61 | 1, 2, 3, 4, 5, 6, 7, 8 | 115256 × 100 | 0.799 | 1.119 |
| Target adaptation | 16310 | 8 | 9 | 16310 × 100 | 0.800 | 1.113 |
| Target test | 14556 | 8 | 10 | 14556 × 100 | 0.799 | 1.122 |

### 4.2 SOH prediction

As shown in Table 3, the proposed method achieved an Average RMSE of 0.009278 and a Macro RMSE of 0.008216. Its MAE was 0.005733, and its MAPE was 0.614%. The source manuscript describes the Average RMSE as approximately 0.95% SOH. The model thus maintained a low overall error on a new batch excluded from training and checkpoint selection. Figure 10 shows that most Batch 10 trajectories followed the observed SOH decline. Tracking was particularly close at high SOH and during intermediate degradation. These results support the complementary roles of the low-order prior and residual temporal module. The model does not rely entirely on direct neural fitting of the complete SOH curve.

![Batch 10 per-cell SOH prediction curves](zjc8.4_SOH_prediction_English_assets/image122.png)

**Fig. 10.** Batch 10 per-cell SOH prediction curves.

**Table 3. SOH prediction metrics**

| Battery index | *n* | RMSE | MAE | MAPE (%) |
|---:|---:|---:|---:|---:|
| 69 | 1702 | 0.011899 | 0.007875 | 0.845 |
| 70 | 1697 | 0.006143 | 0.004690 | 0.495 |
| 71 | 1848 | 0.006891 | 0.004697 | 0.503 |
| 72 | 1811 | 0.005218 | 0.003429 | 0.369 |
| 73 | 2030 | 0.006332 | 0.003909 | 0.423 |
| 74 | 2285 | 0.016069 | 0.011457 | 1.226 |
| 75 | 1783 | 0.004801 | 0.003201 | 0.344 |
| 76 | 1400 | 0.008379 | 0.005268 | 0.565 |
| Average | 14556 | 0.009278 | 0.005733 | 0.614 |
| Macro mean | 8 | 0.008216 | 0.005566 | 0.596 |

Figure 11 shows the cell-level RMSE and MAE distributions. Except for cell 10-6, all test cells had RMSE below 1.20% SOH and MAE no greater than 0.79% SOH. Cell 10-7 had the lowest errors, with RMSE of 0.48% SOH and MAE of 0.32% SOH. In contrast, cell 10-6 had RMSE of 1.61% SOH and MAE of 1.15% SOH. This cell was the most difficult case in Batch 10. The model maintained low cell-level errors for most blind-test cells, but individual degradation behavior still affected isolated cases. Overall, low Average RMSE was accompanied by relatively stable regression across most cells.

![Per-cell RMSE and MAE distribution](zjc8.4_SOH_prediction_English_assets/image123.png)

**Fig. 11.** Per-cell RMSE and MAE distribution for the Batch 10 blind-test cells.

### 4.3 Comparison with recent methods

We reproduced five recent SOH estimation methods for comparison under the same cross-batch setting. PINN4SOH formulates SOH estimation as degradation-dynamics learning constrained by a physics-informed neural network [30]. SSF-WL uses weak-label self-supervision and a Transformer encoder to improve charging-curve representations [31]. AM-MFF applies attention to fuse multisource degradation features [32]. PIKAN constructs cross-domain latent health features through an autoencoder and adversarial learning. It then assigns SOH by nearest-neighbor similarity [33].

Table 4 shows that the proposed method achieved Average RMSE, Macro RMSE, and MAE values of 0.009278, 0.008216, and 0.005733, respectively. Its MAPE was 0.614%. Its RMSE was lower than PINN4SOH, SSF-WL, and AM-MFF by approximately 0.006659, 0.009427, and 0.006333, respectively. The corresponding MAE reductions were approximately 0.005432, 0.007487, and 0.005174. Under the present protocol, generic physical constraints, weak-label representations, or multifeature attention alone did not match the full framework. Combining the low-order prior, residual temporal learning, and degradation-aware alignment yielded lower overall errors in these reproduced comparisons.

Figure 12 visualizes the RMSE and MAE comparison. The proposed method had the lowest overall errors among the evaluated methods. PIKAN's error increased after adapting its relaxation-voltage input to a fixed-length charging-voltage trajectory. This result indicates sensitivity to input representation and target-domain alignment quality under the present reproduction protocol.

**Table 4. Reproduced comparison with recent journal SOH estimation methods under the same cross-batch setting**

| Method | RMSE | Macro RMSE | MAE | MAPE (%) |
|---|---:|---:|---:|---:|
| Proposed | 0.9278 | 0.8216 | 0.5733 | 0.6143 |
| PINN4SOH | 1.5937 | 1.5088 | 1.1165 | 1.1735 |
| SSF-WL | 1.8705 | 1.8080 | 1.3220 | 1.4134 |
| AM-MFF | 1.5611 | 1.5007 | 1.0907 | 1.1540 |
| PIKAN | 7.5347 | 7.3958 | 5.4650 | 5.8582 |
| MSWaveFuser | 1.1480 | 1.0492 | 0.8209 | 0.8606 |

![Comparison with representative recent SOH estimation methods](zjc8.4_SOH_prediction_English_assets/image124.png)

**Fig. 12.** Performance comparison between the proposed method and representative recent SOH estimation methods.

### 4.4 Comparison with baseline models

We compared the proposed method with CNN-LSTM, LSTM, GRU, PCA-GPR, PCA-RBF-SVR, PCA-LinearSVR, and the cubic degradation baseline. Figure 13 summarizes the results. The proposed model achieved Average RMSE of 0.009278, Macro RMSE of 0.008216, and MAE of 0.005733. These values were lower than those of all seven baselines. CNN-LSTM produced an RMSE of 0.016075. Thus, convolutional and recurrent end-to-end modeling alone was less effective under the present cross-batch protocol.

PCA-GPR and PCA-RBF-SVR can perform nonlinear regression on low-dimensional statistical features. However, their representation of local charging-voltage morphology remained limited in this comparison. The cubic degradation baseline produced an RMSE of 0.032649, which was higher than that of the full model. The low-order prior therefore cannot replace temporal residual learning. Its bounded role is to provide a global degradation reference. LSTM and GRU produced RMSE values of 0.032742 and 0.033109, respectively. Without the prior, local temporal aggregation, and target-development alignment, recurrent models showed larger cross-batch errors. The overall advantage therefore reflects functional complementarity rather than a single component.

![Main Batch 10 baseline comparison](zjc8.4_SOH_prediction_English_assets/image125.png)

**Fig. 13.** Main Batch 10 baseline comparison.

### 4.5 Limited-target-data experiments

We evaluated how the amount of target-domain data affected cross-batch SOH prediction. The source-only setting trained on Batch 1–8 and did not use Batch 9 cells for feature alignment. Other settings used 1, 2, 4, or 8 Batch 9 cells as target-development samples.

Source-only training produced an RMSE of 0.010805, whereas using eight Batch 9 cells produced an RMSE of 0.009448, as shown in Fig. 14. Limited target-development data therefore provided a modest benefit to cross-batch adaptation in this setting. However, error did not decrease monotonically with the number of target cells. Using two target cells produced an RMSE of 0.011190, which was higher than source-only training. The benefit of target data thus depended on cell representativeness and degradation-stage coverage, not only sample count. Batch 9 informed model selection and feature alignment, but subset composition affected stability under limited data.

The model contained approximately 31,466 trainable parameters. Its checkpoint size was approximately 0.391–0.392 MB, and inference latency was approximately 0.046 ms per sample. Under the current experimental environment, the framework therefore had low inference overhead while retaining a small model size.

![Target-development budget sensitivity](zjc8.4_SOH_prediction_English_assets/image126.png)

**Fig. 14.** Target-development budget sensitivity.

### 4.6 Ablation experiments

We evaluated the contributions of the retained modules through ablation. The experiments removed the low-order nonlinear prior, attention-assisted temporal aggregation, or MMD regularization. A further control replaced exponential degradation-aware MMD with static MMD.

The full model produced an RMSE of 0.009499 in this ablation experiment, as shown in Fig. 15. Removing the low-order prior increased RMSE to 0.020716 and MAE to 0.013759. This was the largest deterioration among the ablations and indicates that the prior provides key structural support for cross-batch prediction. Removing attention-assisted temporal aggregation increased RMSE to 0.011677 and MAE to 0.008145. Attention therefore improved local voltage-feature aggregation and made a bounded auxiliary contribution to the residual branch.

The no-MMD, static-MMD, and degradation-aware MMD variants produced RMSE values of 0.009677, 0.009635, and 0.009499, respectively. These differences were small. MMD therefore provided only a limited improvement in predictive accuracy within this ablation run.

![Ablation and static-MMD control](zjc8.4_SOH_prediction_English_assets/image127.png)

**Fig. 15.** Ablation and static-MMD control.

We also divided each Batch 10 cell into early, middle, and late stages according to normalized cycle position. Figure 16 reports the stage-wise results. The full model produced RMSE values of 0.003120, 0.009111, and 0.013341 in the early, middle, and late stages, respectively. Removing the low-order prior increased late-stage RMSE to 0.031926. This change supports the prior's role in stabilizing late-stage predictions under the evaluated protocol.

![Stage-wise RMSE comparison](zjc8.4_SOH_prediction_English_assets/image128.png)

**Fig. 16.** Stage-wise RMSE comparison.

## 5 Discussion

We evaluated sensitivity to random initialization using seeds 7, 42, and 123. Figure 17 summarizes the multi-seed results. Removing the low-order prior produced a mean RMSE of 0.022364, which was markedly higher than the full and other ablated models. This result further supports the prior's key structural role. The full model had a three-seed mean RMSE of 0.010381, whereas no-MMD produced 0.010249. The difference was small, and no-MMD was slightly lower. MMD was therefore not a robust source of predictive-accuracy improvement under the current evidence.

![Multi-seed stability](zjc8.4_SOH_prediction_English_assets/image129.png)

**Fig. 17.** Multi-seed stability.

Although MMD contributed little to RMSE, representation diagnostics indicated an alignment effect between the source and target-development domains. Figure 18 visualizes fused SOH features from source Batch 1–8 and target-development Batch 9. The diagnostics include feature MMD², domain-classifier AUC, source-target centroid-gap contributions along principal components, and a normalized composite domain-difference index. Without MMD, source and target-development features retained a visible centroid gap in three-dimensional PCA space. This separation indicates that the residual temporal representation still contained batch-related information. With degradation-aware MMD, target features moved closer to the principal source-feature manifold. Domain-classifier separability decreased concurrently. These observations are consistent with reduced cross-batch mean-embedding differences and less domain-identifying information unrelated to SOH regression.

The principal-component contributions further show that MMD affected both the distance measure and the distribution of the source-target centroid offset. This redistribution supports interpreting MMD as a representation-alignment constraint rather than a dominant predictor of accuracy. It may provide a more consistent input distribution for residual correction, although the present RMSE evidence remains mixed.

![Source-target representation diagnostic](zjc8.4_SOH_prediction_English_assets/image130.png)

**Fig. 18.** Source-target representation diagnostic of the SOH transfer model. (a) Dual-axis comparison of feature MMD² and domain-classifier AUC for the final model and ablated variants. (b) PC1–PC3 contributions to the source-target centroid gap for the final and no-MMD models. (c) Three-dimensional PCA visualization of fused SOH features for the no-MMD and proposed models. Blue points denote source samples, orange points denote Batch 9 target-development samples, and dashed lines denote source-target centroid gaps.

## 6 Conclusion

This study addressed distribution shifts, inconsistent degradation-stage coverage, and cell-specific deviations in cross-batch lithium-ion battery SOH prediction. We proposed a degradation-aware physics-guided hybrid transfer learning framework. Its physical-residual strategy combines a low-order nonlinear degradation prior with a residual temporal module. The prior supplies a shared global reference, while the residual branch learns local cell-specific deviations. Degradation-state-guided curve attention conditions temporal aggregation on estimated degradation state and local charging-voltage morphology. Lifecycle-indexed degradation-aware MMD provides an additional source-target feature-alignment constraint during training.

The framework maintained low overall prediction errors on the evaluated lithium-ion battery dataset and stable regression for most blind-test cells. Ablation results identified the low-order prior as the main structural support and attention as an effective auxiliary representation component. MMD had only a limited and non-robust effect on RMSE, although representation diagnostics supported its alignment role. The current evidence is restricted to cross-batch splits within one dataset and does not establish generalization across battery types. Future work should evaluate multiple datasets, operating conditions, and more complete degradation-stage coverage before broader applicability is claimed.

---

## Author confirmation required

The following items should be checked before journal submission. They are editorial flags and are not part of the manuscript text.

1. **Target journal and format.** No target journal, abstract word limit, or section template was specified. The English is journal-neutral and Nature-leaning, but it has not been conformed to a particular guide for authors.

2. **Canonical MMD module name.** The Chinese source alternates among “生命周期指数感知 MMD”, “生命周期指数退化感知 MMD”, and shorter degradation-aware forms. This draft standardizes the term as **lifecycle-indexed degradation-aware MMD regularization**. Please confirm the intended official name.

3. **Dataset provider name.** Please verify that “Huazhong Artificial Intelligence Lab” is the provider's official English name and add the dataset citation or URL if required.

4. **Table cross-reference.** The source dataset paragraph refers to “Table 1”, but the split summary is labeled Table 2. This draft uses Table 2.

5. **Ambiguous figure cross-reference.** The source text immediately before Eq. (4) says “从图4发现”, although Fig. 4 is the coefficient landscape. The translation states the model decomposition directly and omits that ambiguous cross-reference.

6. **RMSE conversion.** Average RMSE = 0.009278 corresponds to 0.9278% on a unit SOH scale, while the source describes it as approximately 0.95% SOH. Please confirm the intended rounding.

7. **Table 4 units.** Table 4 reports RMSE, Macro RMSE, and MAE on an apparent percentage scale, while Table 3 and the Results text use fractional values. Please add explicit units or rescale one presentation consistently.

8. **Recent-method description.** The text states that five recent methods were reproduced, but it describes only PINN4SOH, SSF-WL, AM-MFF, and PIKAN. Table 4 also includes MSWaveFuser. Please add its description and citation.

9. **PIKAN comparison protocol.** PIKAN was adapted from relaxation-voltage features to fixed-length charging-voltage trajectories. Please confirm that this reproduction is sufficiently comparable and document the adaptation settings.

10. **Different full-model RMSE values.** The manuscript reports 0.009278 in the main evaluation, 0.009448 with eight Batch 9 cells, 0.009499 in ablation, and 0.010381 as the three-seed mean. Please label the seed, split, checkpoint, and evaluation protocol for each value.

11. **Cell identifiers.** Table 3 uses battery indices 69–76, while the narrative uses cell IDs 10-1 to 10-8. Please provide or verify the mapping, especially for cells 10-6 and 10-7.

12. **Figure 8 interpretation.** The source connects the late active training window with higher degradation. Please confirm how training-batch degradation composition supports this stage interpretation.

13. **Equation source format.** The DOCX stores all equations and inline symbols as MathType/OLE graphics. They are preserved exactly as PNG images here. Please provide editable MathType or LaTeX equations for final typesetting.

14. **Abstract evidence.** The source abstract contains no quantitative result. Confirm whether the target journal expects the principal RMSE and test protocol in the abstract.

## Terminology ledger

| Canonical English term | Use in this draft |
|---|---|
| state of health (SOH) | Written in full at first use, then “SOH” |
| cross-batch SOH prediction | Used consistently for the task |
| physical-residual SOH modeling strategy | Overall decomposition into baseline and residual |
| low-order nonlinear degradation prior | Shared global degradation reference, not a complete electrochemical model |
| residual temporal module | Learns cell-specific local deviations around the reference |
| degradation-state-guided curve attention for temporal aggregation | Auxiliary representation and aggregation component |
| lifecycle-indexed degradation-aware MMD regularization | Training-stage source-target alignment regularizer |
| source domain | Batch 1–8 training data |
| target-development domain | Batch 9 adaptation and model-selection data |
| final blind-test set | Batch 10 only; excluded from training and checkpoint selection |

*References were intentionally not translated. All in-text citation numbers were retained from the source manuscript.*
