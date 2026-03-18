<p align="center">
  <a href="https://x-raydar.info"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logo-light.png"><source media="(prefers-color-scheme: light)" srcset="assets/logo-dark.png"><img src="assets/logo-dark.png" alt="X-Raydar" width="320" /></picture></a>
</p>

<h3 align="center">X-Raydar — Radiology Report Classifier</h3>

<p align="center">
  <a href="https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00218-2/fulltext"><img src="https://img.shields.io/badge/Paper-The_Lancet_Digital_Health-blue" alt="Paper" /></a>
  <a href="https://huggingface.co/dnamodel/xraydar-nlp"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model_Weights-yellow" alt="HuggingFace" /></a>
  <a href="https://x-raydar.info"><img src="https://img.shields.io/badge/Website-x--raydar.info-green" alt="Website" /></a>
</p>

---

NLP component of [X-Raydar](https://x-raydar.info), from ["Development and validation of open-source deep neural networks for comprehensive chest x-ray reading"](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00218-2/fulltext) (Cid, Macpherson et al., *The Lancet Digital Health*, 2024).

A fine-tuned RoBERTa model (RoBERTaX) that classifies free-text radiology reports into **45 finding categories** using multi-label classification.

> **This is not for clinical use.**

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

```python
from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("src/model/robertax_pretrained", exist_ok=True)

shutil.copy(
    hf_hub_download("dnamodel/xraydar-nlp", "nlp/robertax1.0.pt"),
    "src/model/robertax1.0.pt"
)

for f in ["pytorch_model.bin", "config.json", "vocab.json", "merges.txt"]:
    shutil.copy(
        hf_hub_download("dnamodel/xraydar-nlp", f"nlp/pretrained/{f}"),
        f"src/model/robertax_pretrained/{f}"
    )
```

### 3. Run inference

```python
import predict

model, tokenizer = predict.build_model()

report = "The heart is enlarged. There is a small left pleural effusion."
input_ids, attention_masks = predict.doc_to_torch([report], tokenizer)
predictions = predict.main(input_ids, attention_masks, model)
print(predictions)
```

Demo reports are provided in `demo_data/`.

## Project Structure

```
src/
├── predict.py                  # Inference (build_model, doc_to_torch, main)
├── model/
│   ├── robertax1.0.pt          # Place fine-tuned weights here
│   └── robertax_pretrained/    # Place pretrained base + tokenizer here
│       ├── pytorch_model.bin
│       ├── config.json
│       ├── vocab.json
│       └── merges.txt
└── inference_script.ipynb      # Example notebook
```

## Requirements

- Python 3.8+
- PyTorch
- transformers
- simpletransformers==0.50.0

See `requirements.txt` for pinned versions.

## Related

| | Repository | Weights |
|-|-----------|---------|
| **X-ray classifier** | [gmontana/xraydar-cv](https://github.com/gmontana/xraydar-cv) | [dnamodel/xraydar-cv](https://huggingface.co/dnamodel/xraydar-cv) |

## Citation

```bibtex
@article{cid2024development,
  title={Development and validation of open-source deep neural networks for
         comprehensive chest x-ray reading: a retrospective, multicentre study},
  author={Cid, Yan Digilov and Macpherson, Matt and Gervais-Andre, Luc and
          Zhu, Yinghui and Franco, Guillermo and Santeramo, Ruggiero and
          Mudali, Divya and Wood, Orlando and Montague, Eoin and Wei, Jiefei and
          others},
  journal={The Lancet Digital Health},
  volume={6}, number={1}, pages={e44--e57},
  year={2024}, publisher={Elsevier},
  doi={10.1016/S2589-7500(23)00218-2}
}
```

## License

Academic research and non-commercial evaluation only. See [LICENSE](LICENSE) for full terms.

## Contact

Giovanni Montana — [g.montana@warwick.ac.uk](mailto:g.montana@warwick.ac.uk) · Commercial licensing — [ventures@warwick.ac.uk](mailto:ventures@warwick.ac.uk)
