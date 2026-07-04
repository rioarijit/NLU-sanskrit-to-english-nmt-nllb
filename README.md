# Sanskrit-to-English NMT — Fine-tuned NLLB-200

Fine-tunes Meta AI's pretrained **facebook/nllb-200-distilled-600M** model on a Sanskrit-English parallel corpus. NLLB-200 is pretrained on 200 languages, including Sanskrit (`san_Deva`) and English (`eng_Latn`), so this notebook adapts existing multilingual knowledge to the target dataset rather than training a translation model from scratch.

**Disclosure:** This notebook uses a pretrained model (`facebook/nllb-200-distilled-600M`, Meta AI, license `cc-by-nc-4.0`), downloaded via the HuggingFace `transformers` library and fine-tuned only on the dataset described below.

## Requirements

- Google Colab with GPU runtime (Runtime → Change runtime type → GPU)
- Packages installed automatically in the first code cell: `transformers`, `sentencepiece`, `accelerate`, `bert-score` (`torch`, `pandas` ship with Colab already)
- Internet access (downloads ~2.5GB of model weights on first run)

## How to run

1. Open the notebook in Google Colab.
2. Upload the `nmt_data` folder (containing the six CSV files listed below) into the Colab file browser, next to the notebook.
3. Runtime → Run all.
4. Outputs: `submission.csv` (predicted translations), printed BLEU/BERTScore, training curve plot, and a final summary block.

## Dataset format

The notebook expects a folder called `nmt_data` with six CSV files:

```
nmt_data/
  train_sa_10000.csv   train_en_10000.csv
  dev_sa_1000.csv      dev_en_1000.csv
  test_sa_1000.csv     test_en_1000.csv
```

Each `_sa_` (Sanskrit) file and its matching `_en_` (English) file share a `Source_id` column used to align sentence pairs, plus a `Sentence_sa` or `Sentence_en` column with the actual text.

## How to change the dataset

All dataset configuration lives in **one cell**: the code cell right after the markdown header `### Dataset Assembly and Loading Data` (cell 8 in the notebook, titled `###Defining data path###`).

To point the notebook at a different dataset, edit these two things inside that cell:

**1. The folder path** — first line of the cell:
```python
dataDirectory="./nmt_data"
```
Change `"./nmt_data"` to wherever your new data folder is.

**2. The filenames** — second line of the cell:
```python
requiredFilesList=["train_sa_10000.csv","train_en_10000.csv","dev_sa_1000.csv","dev_en_1000.csv","test_sa_1000.csv","test_en_1000.csv"]
```
Replace these with your new dataset's filenames. The file-loading calls further down the same cell also reference these names directly and must be updated to match:
```python
trainDf = LoadAndMergePairs(f"{dataDirectory}/train_sa_10000.csv", f"{dataDirectory}/train_en_10000.csv")
devDf   = LoadAndMergePairs(f"{dataDirectory}/dev_sa_1000.csv",   f"{dataDirectory}/dev_en_1000.csv")
testDf  = LoadAndMergePairs(f"{dataDirectory}/test_sa_1000.csv",  f"{dataDirectory}/test_en_1000.csv")
```

**Important:** whatever CSVs you point it at must still have `Source_id`, `Sentence_sa`, and `Sentence_en` columns (or `Sentence_en` only, for a target-language file) — the `LoadAndMergePairs` function (defined earlier in the notebook, under `### Helper Functions and Custom Classes`) expects exactly those column names. If your new dataset uses different column names, rename them in your CSVs first, or edit `LoadAndMergePairs` to match.

**If your new dataset isn't Sanskrit→English:** you'll also need to update the language codes in the cell under `### Initializing the Translation Model`:
```python
srcLangCode = "san_Deva"
tgtLangCode = "eng_Latn"
```
to whichever [FLORES-200 language codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) match your new source/target languages (must be codes NLLB-200 was pretrained on).

## Notes

- Training uses early stopping (patience = 2 epochs) based on dev-set loss, so it may stop before the configured maximum epoch count if the model stops improving.
- Inference time scales with test-set size and available GPU; the 50.7 ms/sentence figure above is specific to a T4 GPU on Colab.
