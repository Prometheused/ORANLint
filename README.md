# ORANLint

> **Restricted evaluation material.** This artifact accompanies “Not on the
> Same Page: Uncovering Specification Inconsistencies in O-RAN Standards,”
> submitted to USENIX Security 2027. Use and redistribution are restricted to
> the Artifact Evaluation Committee and authorized reviewers solely for
> evaluation. See [NOTICE](NOTICE) for the complete terms.

This project implements ORANLint. It covers specification preprocessing,
domain adaptation, synthetic-supervision
construction, sentence-encoder training, candidate mining, NLI screening,
local filtering, and contextual verification.

The project contains source code, configuration, documentation, schemas, and
confirmed findings.

## Safety and credentials

No credential is stored in this directory. OpenAI operations read only the
`OPENAI_API_KEY` process environment variable. The scripts do not accept API
keys on the command line and do not load `.env` files. Preparing requests,
estimating tokens, inspecting status files, and importing modules do not create
an OpenAI client.

Keep credentials outside the repository. For an interactive shell, a key can
be entered without placing its value in shell history:

```sh
read -s -p "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
```

Run `unset OPENAI_API_KEY` after API work.

## Environment

The recorded environment uses Python 3.9.12, PyTorch 2.7.0 with CUDA 12.6,
Transformers 4.52.4, sentence-transformers 5.1.0, and OpenAI SDK 2.17.0. A
multi-GPU CUDA system is required for the full training route.

Install the PyTorch build appropriate for the evaluator's CUDA driver, then
install the remaining pinned dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Access to `meta-llama/Llama-3.1-8B-Instruct` must be approved through Hugging
Face before generator training or synthetic generation. The other public base
models are `BAAI/bge-large-en-v1.5` and `microsoft/deberta-v3-large`.

## Directory layout

Place PDF files under:

```text
data/raw/4G/
data/raw/5G/
data/raw/ORAN/
```

Create these ignored input directories when preparing a fresh checkout:

```sh
mkdir -p data/raw/4G data/raw/5G data/raw/ORAN
```

Generated corpora are written to `data/processed/` and `data/generated/`.
Checkpoints are written to `models/`, execution results to `runs/`, and
training logs to `logs/`. These locations are ignored by Git. JSON schemas for
the principal records are under `schemas/`.

## Confirmed Inconsistencies

`findings/oranlint-confirmed-inconsistencies.csv` contains the confirmed Tier 1
and Tier 2 inconsistencies reported in the paper. Each row identifies the
source documents, pages, sections, paired specification passages, and tier.

The commands below are intentionally shown in execution order. Script names
are descriptive and do not encode that order. Run them from this directory.

Set a run directory once:

```sh
RUN_DIR="runs/default"
mkdir -p "$RUN_DIR"
```

## Full workflow

### Preprocess specifications

Process each corpus independently. The pretraining form preserves paragraph
boundaries and writes flat and hierarchical representations.

```sh
python preprocess_specifications.py --net-type 4G --pretraining \
  --input-dir data/raw/4G --output-dir data/processed/4G

python preprocess_specifications.py --net-type 5G --pretraining \
  --input-dir data/raw/5G --output-dir data/processed/5G

python preprocess_specifications.py --net-type ORAN --pretraining \
  --input-dir data/raw/ORAN --output-dir data/processed/ORAN

python merge_pretraining_corpora.py \
  --oran-jsonl data/processed/ORAN/corpus_ORAN.jsonl \
  --corpus-4g data/processed/4G/corpus_4G.jsonl \
  --corpus-5g data/processed/5G/corpus_5G.jsonl \
  --output-dir data/generated/pretraining
```

### Train the domain generator

The generator uses a LoRA adapter over Llama 3.1 8B Instruct and does not send
training telemetry to an external service.

```sh
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train_domain_generator.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --corpus_path data/generated/pretraining/merged_4G_5G_ORAN.txt \
  --output_dir models/domain_generator \
  --logging_dir logs/domain_generator_training \
  --deepspeed_config configs/deepspeed_generator.json \
  --epochs 5 --seed 42
```

Resume an interrupted Trainer checkpoint with
`--resume-from-checkpoint models/domain_generator/checkpoint-N`.

Generate disjoint sentence-encoder training and validation resources:

```sh
CUDA_VISIBLE_DEVICES=0 python generate_encoder_supervision.py \
  --input_jsonl data/processed/ORAN/corpus_ORAN.jsonl \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-checkpoint models/domain_generator \
  --output_jsonl data/generated/encoder_supervision_train.jsonl \
  --target_records 600 --sample_size 4000 --seed 152

CUDA_VISIBLE_DEVICES=0 python generate_encoder_supervision.py \
  --input_jsonl data/processed/ORAN/corpus_ORAN.jsonl \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-checkpoint models/domain_generator \
  --exclude_jsonl data/generated/encoder_supervision_train.jsonl \
  --output_jsonl data/generated/encoder_supervision_validation.jsonl \
  --target_records 120 --sample_size 4000 --seed 153
```

### Train the sentence encoder

```sh
CUDA_VISIBLE_DEVICES=0 python adapt_sentence_encoder.py \
  --base-model BAAI/bge-large-en-v1.5 \
  --corpus_4g_path data/generated/pretraining/processed_4G/corpus_4G.txt \
  --corpus_5g_path data/generated/pretraining/processed_5G/corpus_5G.txt \
  --corpus_oran_path data/generated/pretraining/processed_ORAN/corpus_ORAN.txt \
  --packed_output_dir data/generated/sentence_encoder_packed \
  --output_dir models/sentence_encoder_adapted \
  --logging_dir logs/sentence_encoder_adaptation \
  --epochs 3 --batch_size 8 --seed 42

CUDA_VISIBLE_DEVICES=0 python train_sentence_encoder.py \
  --train_jsonl data/generated/encoder_supervision_train.jsonl \
  --validation_jsonl data/generated/encoder_supervision_validation.jsonl \
  --model-checkpoint models/sentence_encoder_adapted \
  --output_dir models/sentence_encoder \
  --logging_dir logs/sentence_encoder_training \
  --epochs_mnlr 5 --epochs_triplet 5 \
  --batch_mnlr 8 --batch_triplet 4 \
  --neg_mix_ratio 0.65 --margin 0.3 --fp16 --seed 42
```

Both training phases evaluate against the validation resource. Their metrics
and selected checkpoints are recorded in the training manifest. The checkpoint
used downstream is `models/sentence_encoder/phase_triplet`.

### Mine candidate pairs

```sh
python filter_security_segments.py \
  --corpus data/processed/ORAN/corpus_ORAN.jsonl \
  --keywords configs/security_keywords.txt \
  --output data/generated/security_segments.jsonl

CUDA_VISIBLE_DEVICES=0 python cluster_security_segments.py \
  --jsonl data/generated/security_segments.jsonl \
  --sentence-encoder-checkpoint models/sentence_encoder/phase_triplet \
  --k 20 --pca-dim 100 --device cuda:0 --dtype fp16 \
  --embedding-cache data/generated/security_embeddings.npy \
  --outdir data/generated/clusters

CUDA_VISIBLE_DEVICES=0 python mine_candidate_pairs.py \
  --jsonl data/generated/security_segments.jsonl \
  --cluster_path data/generated/clusters/bge_kmeans_clusters.jsonl \
  --sentence-encoder-checkpoint models/sentence_encoder/phase_triplet \
  --embedding-cache data/generated/security_embeddings.npy \
  --tfidf_min 0.5 --tfidf_max 1.0 --cross-cluster-top-k 1 \
  --device cuda:0 --dtype fp16 \
  --output data/generated/candidate_pairs.jsonl
```

### Train the NLI classifier

Adapt DeBERTa to the O-RAN corpus and retain epoch three:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  adapt_nli_model.py \
  --base-model microsoft/deberta-v3-large \
  --corpus_path data/processed/ORAN/corpus_ORAN.jsonl \
  --output_dir models/nli_domain_adapted \
  --block_size 2048 --epochs 5 --stop_after_epoch 3 --retain_epochs 3 \
  --batch_size 2 --eval_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate 2e-5 --warmup_ratio 0.1 --weight_decay 0.01 \
  --seed 42 --bf16 --no-fp16 --disable_early_stopping --no_deepspeed
```

The NLI generator requires a reserved-evaluation JSONL.
Only its identifiers and normalized texts are excluded from supervision.

```sh
python generate_nli_supervision.py prepare \
  --corpus data/processed/ORAN/corpus_ORAN.jsonl \
  --candidates data/generated/candidate_pairs.jsonl \
  --holdout data/generated/reserved_evaluation_pairs.jsonl \
  --output-dir data/generated/nli_supervision \
  --train-per-class 4000 --dev-per-class 1000 --seed 42

CUDA_VISIBLE_DEVICES=0 python generate_nli_supervision.py render \
  --jobs data/generated/nli_supervision/generation_jobs.jsonl \
  --responses data/generated/nli_supervision/generation_responses.jsonl \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-checkpoint models/domain_generator --batch-size 4 --seed 42

python generate_nli_supervision.py finalize \
  --jobs data/generated/nli_supervision/generation_jobs.jsonl \
  --responses data/generated/nli_supervision/generation_responses.jsonl \
  --holdout data/generated/reserved_evaluation_pairs.jsonl \
  --output-dir data/generated/nli_supervision
```

Rejected generations can be prepared explicitly with the `prepare-retry`
subcommand. Construct the selected neutral mixture with:

```sh
python generate_nli_supervision.py prepare-neutral-augmentation \
  --jobs data/generated/nli_supervision/generation_jobs.jsonl \
  --output data/generated/nli_supervision/neutral_augmentation_jobs.jsonl \
  --limit 6000 --accepted-target 1660 --seed 42

CUDA_VISIBLE_DEVICES=0 python generate_nli_supervision.py render \
  --jobs data/generated/nli_supervision/neutral_augmentation_jobs.jsonl \
  --responses data/generated/nli_supervision/neutral_augmentation_responses.jsonl \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-checkpoint models/domain_generator --batch-size 4 --seed 42

python generate_nli_supervision.py finalize-neutral-augmentation \
  --jobs data/generated/nli_supervision/neutral_augmentation_jobs.jsonl \
  --responses data/generated/nli_supervision/neutral_augmentation_responses.jsonl \
  --holdout data/generated/reserved_evaluation_pairs.jsonl \
  --forbidden-split data/generated/nli_supervision/development_pairs.jsonl \
  --output data/generated/nli_supervision/neutral_augmentation_pairs.jsonl \
  --accepted-target 1660

python generate_nli_supervision.py rebuild-neutral-mix \
  --train data/generated/nli_supervision/train_pairs.jsonl \
  --dev data/generated/nli_supervision/development_pairs.jsonl \
  --augmentation data/generated/nli_supervision/neutral_augmentation_pairs.jsonl \
  --output data/generated/nli_supervision/train_pairs_mixed.jsonl \
  --real-neutral 2340 --generated-neutral 1660 --seed 42
```

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_nli_classifier.py \
  --train_jsonl data/generated/nli_supervision/train_pairs_mixed.jsonl \
  --validation_jsonl data/generated/nli_supervision/development_pairs.jsonl \
  --domain-checkpoint models/nli_domain_adapted/retained/epoch-3 \
  --output_dir models/nli_classifier \
  --max_length 2048 --epochs 5 --stop_after_steps 3000 \
  --batch_size 2 --eval_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_ratio 0.1 --weight_decay 0.01 \
  --seed 42 --bf16 --no-fp16 --no_deepspeed \
  --augment_reverse --require_document_disjoint
```

Development accuracy, precision, recall, macro F1, and loss are written to the
training manifest. Use `--resume-from-checkpoint` to continue an interrupted
Trainer checkpoint.

### Evaluate checkpoints

The standalone evaluators do not modify checkpoints:

```sh
CUDA_VISIBLE_DEVICES=0 python evaluate_sentence_encoder.py \
  --sentence-encoder-checkpoint models/sentence_encoder/phase_triplet \
  --validation-jsonl data/generated/encoder_supervision_validation.jsonl \
  --output "$RUN_DIR/sentence_encoder_evaluation.json"

CUDA_VISIBLE_DEVICES=0 python evaluate_nli_classifier.py \
  --nli-checkpoint models/nli_classifier/checkpoint-3000 \
  --development-jsonl data/generated/nli_supervision/development_pairs.jsonl \
  --batch-size 8 --max-length 2048 --device cuda:0 \
  --output "$RUN_DIR/nli_evaluation.json"
```

The sentence-encoder evaluation reports cosine separation, ranking accuracy,
and triplet-margin accuracy. The NLI evaluation reports loss, accuracy, macro
precision, recall, F1, per-class metrics, and the confusion matrix. All labels
in these resources are generated supervision rather than human annotations.

### Screen and locally filter candidates

```sh
CUDA_VISIBLE_DEVICES=0 python screen_candidate_pairs.py \
  --input data/generated/candidate_pairs.jsonl \
  --nli-checkpoint models/nli_classifier/checkpoint-3000 \
  --output "$RUN_DIR/nli_predictions.jsonl" \
  --selected-output "$RUN_DIR/nli_selected.jsonl" \
  --selection-policy top_k_argmax_contradiction --max-selected 10000 \
  --batch_size 8 --device cuda:0 --dtype fp16

python apply_variant_filter.py \
  --input "$RUN_DIR/nli_predictions.jsonl" \
  --corpus data/processed/ORAN/corpus_ORAN.jsonl \
  --output_prefix "$RUN_DIR/nli_selected" \
  --prediction_field deberta_selected --prediction_value true

python apply_measurement_filter.py \
  --input "$RUN_DIR/nli_selected_for_gpt.jsonl" \
  --output-prefix "$RUN_DIR/verification_queue"
```

Local-rule explanations remain in separate outputs and are not included in
the contextual-verifier prompt.

### Run contextual verification

Preparation is local and makes no API request:

```sh
python run_contextual_verification.py prepare \
  --queue "$RUN_DIR/verification_queue_measurement_filtered_for_gpt.jsonl" \
  --corpus data/processed/ORAN/corpus_ORAN.jsonl \
  --hierarchical data/processed/ORAN/corpus_ORAN_hierarchical.json \
  --run-root "$RUN_DIR/contextual_verification" --workers 8

python run_contextual_verification.py status \
  --run-root "$RUN_DIR/contextual_verification"
```

Review the generated manifests and cost estimate. Only then submit requests:

```sh
python run_contextual_verification.py execute \
  --run-root "$RUN_DIR/contextual_verification" --poll-seconds 900

python run_contextual_verification.py finalize \
  --run-root "$RUN_DIR/contextual_verification"
```

The coordinator does not automatically resubmit unresolved requests. It writes
the final verdicts, inconsistent-verdict queue, unresolved results, retry
candidates, and aggregate summary separately.

## Checkpoint paths

Every model-consuming operation accepts an explicit checkpoint path:

```sh
GENERATOR_BASE="/path/to/llama-base-or-compatible-checkpoint"
GENERATOR_ADAPTER="/path/to/generator-adapter"
SENTENCE_ENCODER="/path/to/trained-sentence-encoder"
NLI_DOMAIN="/path/to/domain-adapted-deberta"
NLI_CLASSIFIER="/path/to/trained-nli-classifier"
```

- Pass `--base-model "$GENERATOR_BASE" --adapter-checkpoint "$GENERATOR_ADAPTER"`
  to either synthetic-generation script.
- Pass `--model-checkpoint "$SENTENCE_ENCODER"` to supervised encoder training,
  or `--sentence-encoder-checkpoint "$SENTENCE_ENCODER"` to clustering and mining.
- Pass `--domain-checkpoint "$NLI_DOMAIN"` to NLI training.
- Pass `--nli-checkpoint "$NLI_CLASSIFIER"` to candidate screening.

Using a trained sentence encoder and NLI classifier permits execution from
candidate mining onward without retraining. Adapter checkpoints must be
compatible with their corresponding base models. Classifier checkpoints must
include the tokenizer and the entailment, neutral, and contradiction label
mapping.
