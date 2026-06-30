# EAGLE-360: Embodied Active Global-to-Local Exploration in 360°

<div align="center">
  <b>Jingtao Xu</b><sup>1</sup>, <b>Zizhuo Lin</b><sup>1</sup>, <b>Jianwen Sun</b><sup>2</sup>, <b>Yi Yang</b><sup>1</sup>, <b>Yawei Luo</b><sup>1*</sup>
  <br>
  <sup>1</sup>Zhejiang University &nbsp;&nbsp; <sup>2</sup>Central China Normal University
  <br>
  <sup>*</sup>Corresponding author
</div>

<br>

<div align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-coming_soon-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/Sansjudge/eagle360_qwen3vl_grpo"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow.svg" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Sansjudge/eagle360_test"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg" alt="Dataset"></a>
  <a href="#running-evaluation"><img src="https://img.shields.io/badge/Eval-available-blue.svg" alt="Eval"></a>
</div>

<br>

![EAGLE-360 teaser](assets/teaser.png)

## Overview

EAGLE-360 studies active object search in 360° panoramic environments. Given an equirectangular panorama and a target description, the model reasons over the global scene, calls a projection tool to inspect local perspective views, and finally predicts the target azimuth and elevation.

Instead of relying on fragmented local exploration from a fixed initial view, EAGLE-360 follows a global-to-local strategy: it first uses panoramic priors to choose a promising direction, then progressively narrows the search space through multi-turn tool use. The inference stack includes Rolling RoPE support for panoramic images and a vLLM-based evaluator.

![EAGLE-360 pipeline](assets/pipeline.png)

## Highlights

- Global-to-local panoramic exploration for embodied 360° visual search.
- Rolling RoPE support for continuous equirectangular panoramas.
- Multi-turn tool-calling inference with `rotate_and_project_panorama`.
- GRPO training improves search accuracy and exploration efficiency.
- Lightweight evaluation code for reproducing the 50-sample Matterport3D result.

## Results

Quantitative results on the EAGLE-360 test set. `Acc.` is the adaptive spherical bFOV accuracy, `GCD` is mean great-circle distance, and `GCD@50°` is the percentage of predictions within 50°. Top-three performances are highlighted with <span style="background-color:#ffe6e6">red</span>, <span style="background-color:#e8f5e9">green</span>, and <span style="background-color:#e8f0ff">blue</span>.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th>Acc.</th>
      <th>GCD</th>
      <th>GCD @</th>
      <th>Fail</th>
      <th colspan="6">All Directions Acc. (%) ↑</th>
    </tr>
    <tr>
      <th>(%) ↑</th>
      <th>(°) ↓</th>
      <th>50° (%) ↑</th>
      <th>(%) ↓</th>
      <th>Front</th>
      <th>Back</th>
      <th>Left</th>
      <th>Right</th>
      <th>Top</th>
      <th>Bottom</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="11"><b>Proprietary Models</b></td></tr>
    <tr><td>GPT-4o</td><td align="right">7.50</td><td align="right">44.26</td><td align="right">80.00</td><td align="right">6.39</td><td align="right">10.31</td><td align="right">2.70</td><td align="right">6.67</td><td align="right">8.89</td><td align="right">25</td><td align="right">0</td></tr>
    <tr><td>Gemini-2.5-Pro</td><td align="right">20.28</td><td align="right">48.89</td><td align="right">78.61</td><td align="right">18.33</td><td align="right">21.65</td><td align="right">4.05</td><td align="right">25.56</td><td align="right">25.56</td><td align="right" bgcolor="#e8f0ff">50</td><td align="right" bgcolor="#e8f0ff">20</td></tr>
    <tr><td colspan="11"><b>Open-source Models</b></td></tr>
    <tr><td>Gemma-3-4b-it</td><td align="right">1.39</td><td align="right">95.23</td><td align="right">25</td><td align="right">10.83</td><td align="right">5.15</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>Gemma-3-12b-it</td><td align="right">2.78</td><td align="right">88.11</td><td align="right">30.83</td><td align="right">12.78</td><td align="right">4.12</td><td align="right">0</td><td align="right">0</td><td align="right">6.67</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>InternVL3.5-4b</td><td align="right">4.72</td><td align="right">77.7</td><td align="right">33.06</td><td align="right">4.17</td><td align="right">2.06</td><td align="right">2.70</td><td align="right">3.33</td><td align="right">11.11</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>InternVL3.5-8b</td><td align="right">5.28</td><td align="right">82.22</td><td align="right">30</td><td align="right">3.89</td><td align="right">19.59</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>Qwen2.5-VL-7B-Instruct</td><td align="right">2.78</td><td align="right">101.10</td><td align="right">22.60</td><td align="right">20.28</td><td align="right">9.28</td><td align="right">0</td><td align="right">1.11</td><td align="right">0</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>Qwen3-VL-4B-Instruct</td><td align="right">8.33</td><td align="right">54.89</td><td align="right">57.22</td><td align="right">6.11</td><td align="right">15.46</td><td align="right">1.35</td><td align="right">4.44</td><td align="right">11.11</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>Qwen3-VL-8B-Instruct</td><td align="right">8.33</td><td align="right">54.72</td><td align="right">69.17</td><td align="right">13.06</td><td align="right">20.62</td><td align="right">4.05</td><td align="right">3.33</td><td align="right">4.44</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td colspan="11"><b>Fine-tuned Models</b></td></tr>
    <tr><td>HVS-3B</td><td align="right">11.11</td><td align="right">41.77</td><td align="right">77.54</td><td align="right">7.22</td><td align="right">18.56</td><td align="right">10.81</td><td align="right">7.87</td><td align="right">7.78</td><td align="right">0</td><td align="right">0</td></tr>
    <tr><td>EAGLE-360 (w/o FOV)</td><td align="right">39.44</td><td align="right">17.54</td><td align="right">94.02</td><td align="right" bgcolor="#e8f5e9">1.11</td><td align="right" bgcolor="#e8f0ff">48.45</td><td align="right">28.38</td><td align="right">41.11</td><td align="right">36.67</td><td align="right" bgcolor="#e8f0ff">50</td><td align="right" bgcolor="#ffe6e6"><b>40</b></td></tr>
    <tr><td>EAGLE-360 (w/o RoPE Rolling)</td><td align="right" bgcolor="#e8f0ff">46.01</td><td align="right" bgcolor="#e8f5e9">16.15</td><td align="right" bgcolor="#e8f0ff">94.72</td><td align="right">2.7</td><td align="right">46.18</td><td align="right" bgcolor="#e8f5e9">45.65</td><td align="right" bgcolor="#e8f5e9">45.25</td><td align="right" bgcolor="#e8f5e9">47.15</td><td align="right" bgcolor="#e8f5e9">60</td><td align="right" bgcolor="#e8f5e9">31.58</td></tr>
    <tr><td>EAGLE-360 (w/o GRPO)</td><td align="right" bgcolor="#e8f5e9">46.94</td><td align="right" bgcolor="#ffe6e6"><b>14.03</b></td><td align="right" bgcolor="#ffe6e6"><b>97.22</b></td><td align="right" bgcolor="#ffe6e6"><b>0.83</b></td><td align="right" bgcolor="#e8f5e9">55.67</td><td align="right" bgcolor="#e8f0ff">41.89</td><td align="right" bgcolor="#e8f0ff">43.44</td><td align="right" bgcolor="#e8f0ff">44.44</td><td align="right" bgcolor="#ffe6e6"><b>75</b></td><td align="right" bgcolor="#ffe6e6"><b>40</b></td></tr>
    <tr><td><b>EAGLE-360</b></td><td align="right" bgcolor="#ffe6e6"><b>64.44</b></td><td align="right" bgcolor="#e8f0ff">16.89</td><td align="right" bgcolor="#e8f5e9">96.12</td><td align="right" bgcolor="#e8f0ff">2.05</td><td align="right" bgcolor="#ffe6e6"><b>72.16</b></td><td align="right" bgcolor="#ffe6e6"><b>60.81</b></td><td align="right" bgcolor="#ffe6e6"><b>55.56</b></td><td align="right" bgcolor="#ffe6e6"><b>71.11</b></td><td align="right" bgcolor="#ffe6e6"><b>75</b></td><td align="right" bgcolor="#ffe6e6"><b>40</b></td></tr>
  </tbody>
</table>

For the lightweight public eval split in this repository, our local reproduction with `eval.py` and the step-1000 checkpoint reaches **31/50 = 62.0%** bFOV accuracy.

## Getting Started

### Installation

```bash
git clone https://github.com/Sansju/EAGLE-360.git
cd EAGLE-360
bash install.sh
```

The script installs `vllm==0.11.0`, `transformers==4.57.0`, `torch==2.8.0`, and applies the panoramic Rolling RoPE patches.

For manual installation:

```bash
pip install -r requirements.txt

VLLM_DIR=$(python -c "import vllm,os; print(os.path.dirname(vllm.__file__))")
TRANS_DIR=$(python -c "import transformers,os; print(os.path.dirname(transformers.__file__))")
cp -r patches/vllm/* $VLLM_DIR/
cp -r patches/transformers/* $TRANS_DIR/
```

Do not pre-install a different PyTorch CUDA wheel before installing vLLM. `vllm==0.11.0` pins `torch==2.8.0` with CUDA 12.8 dependencies on Linux. `flash-attn` is not required for evaluation; if an old `flash-attn` wheel fails to import with PyTorch 2.8, uninstall it or rebuild it for the same PyTorch/CUDA stack.

### Data

The public EAGLE-360 test split is available on Hugging Face: [Sansjudge/eagle360_test](https://huggingface.co/datasets/Sansjudge/eagle360_test). It contains `test.json` and a flat `images/` directory with 354 renamed panoramic images for 360 test samples.

Download it with:

```bash
huggingface-cli download Sansjudge/eagle360_test \
  --repo-type dataset \
  --local-dir ./eagle360_test
```

`eval.py` resolves images as:

```text
{pano_dir}/{basename(panoramic_image)}
```

For the Hugging Face release, pass `--test_file ./eagle360_test/test.json` and `--pano_dir ./eagle360_test/images`.

### Model Weights

The EAGLE-360 Qwen3-VL GRPO checkpoint is available on Hugging Face: [Sansjudge/eagle360_qwen3vl_grpo](https://huggingface.co/Sansjudge/eagle360_qwen3vl_grpo).

Download it with:

```bash
huggingface-cli download Sansjudge/eagle360_qwen3vl_grpo \
  --local-dir ./checkpoints/hf_merged
```

You can also pass the local checkpoint path directly to `--model`.

## Running Evaluation

Quick 50-sample evaluation:

```bash
python eval.py \
  --model ./checkpoints/hf_merged \
  --test_file data/test.json \
  --pano_dir /data/matterport3d \
  --n_samples 50
```

Full evaluation:

```bash
python eval.py \
  --model ./checkpoints/hf_merged \
  --test_file data/test.json \
  --pano_dir /data/matterport3d
```

Results are saved as `eval_{model_name}_{timestamp}.json`.

## Repository Structure

```text
panoramic-360-eval/
├── assets/              # Teaser and pipeline figures
├── data/                # Test metadata and data preparation notes
├── patches/             # vLLM and transformers panoramic patches
├── eval.py              # Main evaluator
├── install.sh           # Environment setup and patch installation
├── requirements.txt
└── upload_model.sh
```

## Citation

```bibtex
@misc{xu2026eagle360,
  title  = {EAGLE-360: Embodied Active Global-to-Local Exploration in 360°},
  author = {Xu, Jingtao and Lin, Zizhuo and Sun, Jianwen and Yang, Yi and Luo, Yawei},
  year   = {2026}
}
```

## Acknowledgements

The implementation builds on [vLLM](https://github.com/vllm-project/vllm), [Transformers](https://github.com/huggingface/transformers), and Qwen3-VL. The patch files in `patches/` follow the licenses of their upstream projects.
