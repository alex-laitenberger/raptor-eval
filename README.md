# Experiment Environment for Evaluating RAPTOR method

This repository contains the full experiment environment necessary to reproduce the experiments on the RAPTOR method (Sarthi et al., 2024) in our paper:

**Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models** (Laitenberger et al., 2025) - [Paper (TODO: add link)](arxiv.org) - [Repo-Overview](https://github.com/Lightnz/stronger-baselines-rag/)

The code is based on the official [RAPTOR Github Repository](https://github.com/parthsarthi03/raptor) commit state `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`. 

We added an experiment environment to run the benchmarks: QuALITY, NarrativeQA and InfinityBench. Additionally we made a few adjustments to the method, to warrant fairness and comparability with the other tested methods. These changes include:
- sentence splitting with NLTK and fix in chunking method
- adjustments for parallel processing in tree creation
- custom QA models to use the same prompts when comparing against other methods

All changes are tracked in the commit history.

## Setup

### Requirements

Before using this repository, ensure Python 3.11+ is installed. You can install the requirements with:

```bash
pip install -r requirements.txt
```

### OpenAI API Key

We used the OpenAI API for our experiments. For reproducing the experiments you need to create a file named `config.py` in the project root directory (next to `requirements.txt`) and add the line `OPENAI_API_KEY = "your-api-key"` with your API key. The file is excluded from repo updates in the `.gitignore`, so don't worry about exposing the key in case you plan to commit.

## Bash commands & Logs

All provided bash commands are to be executed from the project's root directory.
Provided scripts generally create log-files under `experiments/logs/`.

## Experiments

### Datasets

QuALITY: Question Answering with Long Input Texts, Yes! (Pang et al., 2022) - [Paper](https://arxiv.org/pdf/2112.08608) - [Github](https://github.com/nyu-mll/quality?tab=readme-ov-file)

∞bench: Extending long context evaluation beyond 100K tokens. (Zhang et al., 2024) - [Paper](https://arxiv.org/abs/2402.13718) - [Github](https://github.com/OpenBMB/InfiniteBench)

The NarrativeQA reading comprehension challenge (Kočiský et al., 2018) - [Paper](https://arxiv.org/abs/1712.07040) - [Github](https://github.com/google-deepmind/narrativeqa)


### Preparing datasets

- Prepare data folders  
    ```bash
    mkdir -p data/{quality,narrativeqa,infinity_bench}/raw
    ```

#### QuALITY (development set)
- Download and unzip [QuALITY v1.0.1](https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip) 

- Copy the file `QuALITY.v1.0.1.htmlstripped.dev` into `/data/quality/raw`

- Run the preprocessing script to group the dataset by documents.
    ```bash
    python -m data_source.quality.preprocess_quality
    ```
    It should have created a new json file under the `data/quality/preprocessed` path.

#### ∞bench (En.MC)
- Download the [longbook_choice_eng.jsonl](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_choice_eng.jsonl) file directly from huggingface into the `data/infinity_bench/raw` folder.

- Run the preprocessing script:
    ```bash
    python -m data_source.infinity_bench.preprocess_longbook_choice_eng
    ```
    It should have created a new json file under `data/infinity_bench/preprocessed`.

#### NarrativeQA
- Clone the repository, e.g. to your home directory
    ```bash
    cd ~
    git clone git@github.com:google-deepmind/narrativeqa.git
    ```

- Execute the `download_stories.sh` script in the NarrativeQA repository to download the 1572 documents into the `tmp` directory. This might take a while.
    ```bash
    cd ~/narrativeqa #or your clone destination

    bash download_stories.sh
    ```

- Back in the raptor-eval repository run the preprocessing script to prepare the QA experiment.
You might need to change the `NARRATIVE_QA_PATH` variable in case you did not clone the NarrativeQA repository into your home directory. 
    ```bash
    python -m data_source.narrative_qa.preprocess_narrative
    ```
    It should have created the file `processed_qaps_test.json` in the `data/narrativeqa/preprocessed` folder.


### 🌳 Precreate trees

#### 💰 Budget

The budget requirements outlined below are approximate estimates and should be considered as rough guidelines. If you plan to reproduce the experiments, please conduct your own calculations to ensure the costs align with your budget.

The calculcation only considers input tokens to create the summaries for the first layer. An additional amount needs to be added for input tokens to create summaries for higher layers and for output tokens for all created summaries. From our experience the input tokens are a good starting point to approximate the budget level. From there you might want to double it to have an approximate total budget.

| Dataset     | Budget Level     | First Layer Input tokens              |
|-------------|------------------|---------------------------------------|
| QuALITY     |  ~ $0.15         | 115 docs x 5.1k avg tokens < 1M       |
| ∞bench      |  ~ $2 - $4       | 58 docs x 184k avg tokens = 11M       |
| NarrativeQA |  ~ $4 - $8       | 355 docs x 57.7k avg tokens = 20M     |



#### 📝 Scripts
We prepared the scripts `precreate_trees.py` for each dataset to precreate all RAPTOR trees for the experiments. It uses GPT-4o-mini by default and a maximum chunk length of 100 tokens both for document chunking and for creating summaries / abstractive chunks.

To generate embeddings for queries and document chunks we use the [Snowflake Arctic-embed-m 1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) model with 109M parameters (Merrick, 2024). To establish a fair comparison this model is used across all methods in our paper that use embedding-based retrieval.

Upon completion there should be created trees in the output folder under `experiments/artifacts/trees/<dataset>/...`.

Run the scripts with:

##### QuALITY
```bash
python -m experiments_source.quality.precreate_trees
```

##### ∞bench
```bash
python -m experiments_source.infinity_bench.longbook_choice_eng.precreate_trees
```

##### NarrativeQA
You might need to change `NARRATIVEQA_PATH` in the script and adapt it to the location of your cloned NarrativeQA repository with the downloaded documents in the `tmp` folder.
```bash
python -m experiments_source.narrative_qa.precreate_trees
```

### 🚀 Run Experiments

#### 💰 Budget

The budget requirements outlined below are approximate estimates and should be considered as rough guidelines. If you plan to reproduce the experiments, please conduct your own calculations to ensure the costs align with your budget.

The necessary budget depends on the setting of hyperparameters.
If you set `max_tokens` to 1k, you can approximate the used input tokens for the QuALITY development set like this:
`2k questions x 1k max_tokens = 2M total input tokens`
which results in about **$0.30 with GPT4o-mini and $5.00 with GPT4o**.

With the maximum setting this experiment should require approximately:
`2k questions x 6k average tokens per document = 12M total input tokens`
which results in about **$1.80 (GPT4o-mini) and $30 (GPT-4o)**.

The following table concludes the budget level estimates for all datasets across different `max_tokens` settings.


| Dataset     | max_tokens | Budget Level (GPT4o-mini)    | Budget Level (GPT4o) | Approximate Input tokens              |
|-------------|------------|------------------|------------------|---------------------------------------|
| QuALITY     |  1k        | $0.30            | $5               | 2k questions x 1k context = 2M        |
| QuALITY     |  8k        | $2            | $30              | 2k questions x 6k avg context = 12M       |
| ∞bench      |  10k       | $1            | $6               | 229 questions x 10k tokens = 2.3M       |
| ∞bench      |  40k       | $2            | $23              | 229 questions x 40k tokens = 9.2M       |
| NarrativeQA |  10k | $16 | $262 (not executed) | 10.5k questions x 10k tokens = 105M |
| NarrativeQA |  40k | $63 | $1013 (not executed) | 10.5k questions x 40k tokens = 405M |


#### 📝 Scripts

We provide scripts called `run_experiment.py` under `experiments_source/<dataset>/` to run the experiments on the previosuly created trees.

Before executing you need to make a few adjustments in each script:

- the scripts use GPT-4o-mini by default. To conduct experiments with GPT-4o, adjust `OPENAI_MODELSTRING`. The model strings for GPT-4o-mini and GPT-4o are provided as comments and can be easily switched by commenting/uncommenting the desired option.

- change `STORED_TREES_PATH` to match your precreated trees folder. It should look something like `experiments/artifacts/trees/<dataset>/2025-03-12_16-30-raptor-chunk_size_100_Snowflake_Quality_dev` but with your individual time-stamp.

- the scripts use parallelity to run the experiments on multiple documents at the same time. If you want to run the experiment sequentially, or control the amount of parallelity find the line `with ThreadPoolExecutor() as executor:` and adjust it accordingly (e.g., `ThreadPoolExecutor(max_workers=1)` to run sequentially).

- you can set the hyperparameters for the experiment by modifying the experiments list in the run_experiment_batch() function. `max_tokens` determines the number of tokens included in the final retrieved context. `top_k` specifies how many chunks are retrieved initially, before the max_tokens limit is applied. 

In our experiments, we primarily use `max_tokens` as the key parameter to control the size of the retrieved context, as it provides more precise control. Since we preserve sentence boundaries, chunk lengths vary dynamically between 50 and 100 tokens. 
To ensure enough tokens are available before applying the `max_tokens` limit, we set `top_k` accordingly. For example, if max_tokens is 1000, we set top_k to 20. Given chunk sizes of 50–100 tokens, this retrieves an initial context of 1000–2000 tokens, which is then shortened to maximum 1000 tokens (preserving full chunks). 

The scripts output the model's answers into a `jsonl` file under `experiments/artifacts/answers/<dataset>`. 

For full reproducibility, we provide our experiment hyperparameter settings as comments in the script.

Run the scripts with:

##### QuALITY
```bash
python -m experiments_source.quality.run_experiment
```

##### ∞bench
```bash
python -m experiments_source.infinity_bench.longbook_choice_eng.run_experiment
```

##### NarrativeQA
```bash
python -m experiments_source.narrative_qa.run_experiment
```

### 📊 Evaluate

We prepared scripts to evaluate stored answer files conveniently.
You can use them with multiple answer files in the `experiments/artifacts/answers/<dataset>` folder.

Run:

#### QuALITY
```bash
python -m experiments_source.quality.eval
```
#### ∞bench
```bash
python -m experiments_source.infinity_bench.longbook_choice_eng.eval
```

#### NarrativeQA
```bash
python -m experiments_source.narrative_qa.eval
```

#### Results
The scripts create a `json` and `csv` file in the folder of the respective answer files.

E.g., the json-file for QuALITY results looks like this:
```bash
"file": "2025-03-13_13-29-raptor-quality-dev_0_top-k-20_mt-1000_gpt-4o-mini-2024-07-18.jsonl",
        "total_entries": 2086,
        "correct": 1440,
        "accuracy": 69.03,
        "avg_tokens": 1119.77,
        "hard_entries": 1065,
        "hard_correct": 617,
        "hard_accuracy": 57.93,
        "non_hard_entries": 1021,
        "non_hard_correct": 823,
        "non_hard_accuracy": 80.61
```

`avg_tokens`: Since we track the exact amount of input tokens for each evaluated question we can calculate the average input tokens per question, which is a useful metric for the required budget and resulting efficiency.

## References

Alex Laitenberger, Christopher D. Manning and Nelson F. Liu. 2025. Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models [Paper (TODO: add link)](www.arxiv.org) - [Github](https://github.com/Lightnz/stronger-baselines-rag/)

Luke Merrick. 2024. Embedding and clustering your data can improve contrastive pretraining. - [Paper](https://arxiv.org/abs/2407.18887) - [Huggingface](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In Proc. of ICLR. - [Paper](https://arxiv.org/abs/2401.18059) - [Github](https://github.com/parthsarthi03/raptor)

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. QuALITY: Question answering with long input texts, yes! In Proc. of NAACL. - [Paper](https://arxiv.org/abs/2112.08608) - [Github](https://github.com/nyu-mll/quality)

Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA reading comprehension challenge. Transactions of the Asso-361
ciation for Computational Linguistics, 6:317–328. - [Paper](https://arxiv.org/abs/1712.07040) - [Github](https://github.com/google-deepmind/narrativeqa)

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024. ∞bench: Extending long context evaluation beyond 100K tokens. In Proc. of ACL. - [Paper](https://arxiv.org/abs/2402.13718) - [Github](https://github.com/OpenBMB/InfiniteBench)

## License

This project is released under the [MIT License](LICENSE.txt).

It builds upon the original work from [RAPTOR](https://github.com/parthsarthi03/raptor), which is also licensed under the [MIT License](https://github.com/parthsarthi03/raptor/blob/master/LICENSE.txt). Substantial changes have been made to adapt the code for evaluation purposes.

Please refer to the [LICENSE](LICENSE.txt) file for full licensing details.

## Citation

TODO: add bibtex citation here
