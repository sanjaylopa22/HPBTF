# Hierarchical Patch Based Transformer with Learnable Weighted Loss Optimization for Multivariate Time Series Forecasting (HPBTF)
Forecasting, a fundamental task in machine learning, involves predicting future values of a time series based on its historical behavior. This paper introduces a novel Hierarchical Patch Based Transformer with Learnable Weighted Loss Optimization (HPBTF) for multivariate time series forecasting. The proposed model leverages a hierarchical patch embedding strategy to capture both local and global temporal dependencies, significantly reducing sequence length and computational cost while maintaining high forecasting accuracy. Through a modular Transformer encoder equipped with flexible attention and channel-wise normalization, HPBTF efficiently models complex interdependencies among multiple variables. Additionally, the integration of a learnable weighted loss function enhances adaptability across various time series tasks such as forecasting, imputation, classification, and anomaly detection. Experimental results demonstrate that HPBTF outperforms traditional transformer-based models in scalability, versatility, speed up and predictive performance.

# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting**

## Leaderboard for Time Series Analysis as Baseline Models

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/PatchTST_AdvancedPatchEmbed_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/PatchTST_AdvancedPatchEmbed_M4.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)

Or describe it in Issues.

