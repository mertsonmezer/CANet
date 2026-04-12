<h1 align="center">CANet: ChronoAdaptive Network</h1>
<h3 align="center">Long-term Time Series Forecasting</h3>
<p align="center">
  <a href="https://github.com/mertsonmezer/CANet/stargazers"><img src="https://img.shields.io/github/stars/mertsonmezer/CANet" alt="Stars"></a>
  <a href="https://github.com/mertsonmezer/CANet/network/members"><img src="https://img.shields.io/github/forks/mertsonmezer/CANet" alt="Forks"></a>
  <a href="https://github.com/mertsonmezer/CANet/issues"><img src="https://img.shields.io/github/issues/mertsonmezer/CANet" alt="Issues"></a>
  <a href="https://github.com/mertsonmezer/CANet/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mertsonmezer/CANet" alt="License"></a>
</p>
<p align="center">
  <b>Official implementation</b> |
  <a href="https://doi.org/10.1016/j.neunet.2025.108495">Paper (Neural Networks)</a> |
  <a href="https://arxiv.org/abs/2504.17913">arXiv</a> |
  <a href="#citation">Cite</a>
</p>
<p align="center">
  <img src="figs/model_architecture.jpg" alt="CANet Architecture" width="80%"/>
</p>
<p align="center"><i>Figure: CANet's architecture tackles the challenge of non-stationarity in time series data</i></p>

## 🌟 Why CANet?
Real-world time series data is messy. Distribution shifts, temporal changes in statistical properties, and non-stationarity plague traditional forecasting models. While existing approaches struggle with these challenges, CANet thrives. CANet achieves breakthrough performance by:

* 🔄 Preserving non-stationarity through our innovative Style Blending Gate
* 🔍 Dynamically adapting to statistical changes with AdaIN
* 📊 Handling multi-scale patterns via multi-resolution patching
* 🔮 Reducing noise through Fourier analysis-based adaptive thresholding
* ⚡ Optimizing efficiency with Stacked Kronecker Product Layer

## 📈 Impressive Results
Extensive experiments on real-world datasets demonstrate CANet's superiority:

* 42% reduction in Mean Squared Error (MSE)
* 22% reduction in Mean Absolute Error (MAE)

Compared to state-of-the-art methods, CANet consistently delivers more accurate forecasts across diverse domains.

## 🛠️ Setup

**Prerequisites**
- Python ≥ 3.10
- CUDA-capable GPU recommended; CPU training is feasible but slow

**1. Install uv**
If you don't have [uv](https://docs.astral.sh/uv/) installed, follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

**2. Clone the repository**
```bash
git clone https://github.com/mertsonmezer/CANet.git
cd CANet
```

**3. Install dependencies**
```bash
uv sync
```

This creates a virtual environment and installs all required dependencies automatically. All commands below use `uv run` and do not require activating the virtual environment. To activate it manually for ad-hoc use:
```bash
source .venv/bin/activate
```

**4. Verify installation**
```bash
uv run python -c "import canet; print('Environment OK')"
```

## 📊 Datasets

Download the benchmark datasets from:

[![thuml/Time-Series-Library](https://img.shields.io/badge/-Time_Series_Library_(TSLib)-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/thuml/Time-Series-Library)

Place the CSV files under a `datasets/` directory at the repo root, matching the structure below:

```
datasets/
├── ETT-small/       # ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
├── Electricity/     # electricity.csv
├── Exchange/        # exchange_rate.csv
├── ILI/             # illness.csv
├── Traffic/         # traffic.csv
└── Weather/         # weather.csv
```

> **Note on `--root_path`:** experiment scripts pass this path relative to the working directory (repo root). ETT/Electricity/Exchange/Traffic/Weather scripts use bare subdirectory names (e.g., `ETT-small/`); the ILI script uses the full `datasets/ILI/` prefix. Ensure your data layout matches these paths before running.

## 🧪 Experiments
We provide ready-to-use bash scripts for running experiments. For example, to run the long-term forecasting experiment on the ETTh1 dataset:

1. Ensure you have the ETTh1 dataset downloaded and ready.
2. Make the experiment script executable and run it with the following commands:

```bash
chmod +x scripts/etth1.sh  # Ensure the script is executable
bash scripts/etth1.sh      # Run the experiment
```

As shown in the figure below, CANet consistently achieves superior performance on long-term forecasting tasks across various real-world datasets, demonstrating its robustness and effectiveness in handling non-stationary time series data.

<p align="center">
  <img src="figs/longterm_forecasting_results.jpg" alt="Long-term Forecasting Results" width="95%"/>
</p>
<p align="center"><i>Figure: CANet outperforms state-of-the-art models across multiple datasets and forecasting horizons</i></p>

## 📬 Contact
If you have any questions or concerns, feel free to reach out:

📧 **Email:** [mert.sonmezer@metu.edu.tr](mailto:mert.sonmezer@metu.edu.tr)
🐛 **Issues:** [Submit an issue](https://github.com/mertsonmezer/CANet/issues)

## 📖 Citation
If you found this work useful, please consider citing it:
```bibtex
@article{canet2025,
  title     = {CANet: ChronoAdaptive Network for Long-term Time Series Forecasting},
  author    = {Mert Sonmezer and Seyda Ertekin},
  journal   = {Neural Networks},
  year      = {2025}
}
```

## 🙏 Acknowledgements
We appreciate the following papers and their repositories for providing valuable codebases:

| Papers     | Links                                     |
|------------|-------------------------------------------|
| TSLANet: Rethinking Transformers for Time Series Representation Learning | [![emadeldeen24/TSLANet](https://img.shields.io/badge/-TSLANet-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/emadeldeen24/TSLANet) |
| A Time Series is Worth 64 Words: Long-term Forecasting with Transformers | [![yuqinie98/PatchTST](https://img.shields.io/badge/-PathTST-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/yuqinie98/PatchTST)|
| Deep Adaptive Input Normalization for Time Series Forecasting | [![passalis/dain](https://img.shields.io/badge/-DAIN-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/passalis/dain) |

