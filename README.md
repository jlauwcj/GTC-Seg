# GTC-Seg: Cross-Modal Semantic and Temporal Consistency Learning for Weakly-Supervised Point Cloud Semantic Segmentation

> **📢 Notice:** This repository is the official implementation of the paper *"Cross-Modal Semantic and Temporal Consistency Learning for Weakly-Supervised Point Cloud Semantic Segmentation"* (Under Review at **Pattern Recognition**). 
> 
> ⏳ **The full source code, dataset preparation scripts, and pre-trained models are currently undergoing internal cleanup and will be released here upon publication or during the rebuttal phase.**

<p align="center">
  <img src="figs/Figure_1.png" width="100%">
</p>

## 💡 Abstract
Weakly supervised 3D point cloud semantic segmentation (WS3DSS) under extreme sparsity (e.g., 0.1% labels) constitutes a severely ill-posed learning problem. To address the absence of multi-level consistency regularization, we propose **GTC-Seg**, a unified framework that enforces:
1. **Geometric Structural Consistency (LEAP):** Preserves high-frequency topological details before spatial downsampling via explicit relative position encoding.
2. **Cross-Modal Semantic Consistency (CGCM):** Safely injects 2D foundation model (MaskCLIP) priors into 3D feature space through a category-aware rejection gate.
3. **Temporal Prediction Consistency (PLR):** Blocks error accumulation using a dual-scale temporal memory bank and dynamic thresholding.

## 🏗️ Framework Overview

<p align="center">
  <img src="figs/Figure_2.png" width="100%">
</p>

> **Overview of the proposed GTC-Seg framework.** The architecture collaboratively comprises a hybrid 3D backbone for native feature extraction and two parallel consistency enhancement streams: the Cross-modal Gated Consistency Module (CGCM) for external semantic anchoring and the Pseudo-Label Refinery (PLR) for temporal supervisory stabilization.

## 📈 Main Results (Preview)
Under **0.1%** extremely sparse supervision, GTC-Seg effectively overcomes geometric over-smoothing and semantic ambiguity, achieving State-of-the-Art performance:
- **S3DIS (Area 5):** `65.6%` mIoU
- **ScanNet V2:** `65.3%` mIoU (Val) / `65.1%` mIoU (Test)
- **SemanticKITTI:** `53.4%` mIoU

## 🛠️ Codebase Structure (Coming Soon)
The upcoming code release will include:
- [ ] Training and evaluation scripts for S3DIS, ScanNet V2, SemanticKITTI, LiDAR-Net, and STPLS3D.
- [ ] Implementations of the LEAP, CGCM, and PLR modules.
- [ ] Pre-trained weights for quick reproduction.
- [ ] Detailed data preprocessing guidelines.

## 🔗 Citation
If you find our methodology interesting, please consider citing our work:
```bibtex
@article{niu2026gtcseg,
  title={Cross-Modal Semantic and Temporal Consistency Learning for Weakly-Supervised Point Cloud Semantic Segmentation},
  author={Niu, Zhuozhen and Wen, Changji and Zhang, Leyao and Shi, Yaqi and Li, Siyang and Li, Mingqi and Wang, Shengsheng and Lv, Yanfeng and Chen, Hongbing},
  journal={Under Review},
  year={2026}
}
