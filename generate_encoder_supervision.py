# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

import argparse
import hashlib
import os
import random
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

TRUE_FLAG = 2
FALSE_FLAG = 0
MALFORMED_FLAG = 1
MAX_ATTEMPTS = 3
PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = PROJECT_ROOT / "models/domain_generator"
ORAN_JSONL = PROJECT_ROOT / "data/processed/ORAN/corpus_ORAN.jsonl"
SYNTHETIC_OUTPUT = PROJECT_ROOT / "data/generated/encoder_supervision.jsonl"
EXAMPLES_PATH = PROJECT_ROOT / "configs/generation_examples.json"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path):
    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)

class LlamaChat:
    def __init__(self, adapter_path=ADAPTER_PATH, examples_path=EXAMPLES_PATH, base_model=MODEL_NAME):
        # Load LoRA config to get base model path
        self.project_root = PROJECT_ROOT
        self.examples_path = self._resolve_path(examples_path)
        with self.examples_path.open(encoding="utf-8") as f:
            self.examples_json = json.load(f)
        self.adapter_path = self._resolve_path(adapter_path)
        self.peft_model_path = str(self.adapter_path)
        print("Peft model path: ", self.peft_model_path)
        self.config = PeftConfig.from_pretrained(self.peft_model_path)

        print("Before Model loading")

        # Load base model (must be same as used during training)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=None if Path(str(base_model)).exists() else MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded")

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.base_model, self.peft_model_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.peft_model_path)

        with open(os.path.join(self.peft_model_path, "chat_template.jinja")) as f:
            self.tokenizer.chat_template = f.read()


        print("chat template ready")

        # Important fix: set pad token to eos if missing
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()



    def chat(self, message, type = Literal["inconsistent" , "paraphrased" , "randomized"]):

        # example_inconsistent = """Input: "The DU must maintain uplink time synchronization within ±130 ns of the RU."
        # Output: {" inconsistent" : "The required DU-RU uplink time alignment tolerance is ±65 ns."}"""

        # example_paraphrased = """Input: "The Open Fronthaul interface shall use IEEE 1588v2 or SyncE for time synchronization between DU and RU."
        # Output: {"paraphrased" : "The DU shall support frame alignment with the RU according to IEEE 1588v2 (PTP) or Synchronous Ethernet (SyncE)."}"""

        # example_randomized = """Input: "The DU shall support the following O-RAN 7-2x functional split options: Option 7-2x, Option 7-2e, Option 7-2f."
        # Output: {"randomized" : "The DU shall support frame alignment with the RU according to IEEE 1588v2 (PTP) or Synchronous Ethernet (SyncE).}"""

        examples_inconsistent = self.examples_json.get("inconsistent", "")
        examples_paraphrased = self.examples_json.get("paraphrased", "")
        examples_randomized = self.examples_json.get("randomized", "")
        if type == "inconsistent":
            examples = examples_inconsistent
        elif type == "paraphrased":
            examples = examples_paraphrased
        elif type == "randomized":
            examples = examples_randomized
        system_message = f"""You are an expert of ORAN specifications. For the following text, produce a(n) {type} version of the text in **JSON format** while preserving the **original context as closely as possible**. 
        Ensure that the output is a valid sentence in English and adheres to the technical correctness of ORAN specifications. *Do not invent* technical terms.
        Example: 1. For weakly {type}, Input: "{examples["weak"]["text"]}"
        Output: {{"{type}": "{examples["weak"]["output"]}"}}
        2. For moderately {type}, Input: "{examples["moderate"]["text"]}"
        Output: {{"{type}": "{examples["moderate"]["output"]}"}}
        3. For strongly {type}, Input: "{examples["strong"]["text"]}"
        Output: {{"{type}": "{examples["strong"]["output"]}"}}
        """

        #print(system_message)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        print("Input size: ", len(inputs[0]))
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        #print(self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1])

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1]

    @staticmethod
    def _resolve_path(path):
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @staticmethod
    def _normalize_text(text):
        return " ".join(str(text or "").split()).casefold()

    @staticmethod
    def _record_id(record):
        value = record.get("id")
        return None if value is None else str(value)

    def _load_jsonl(self, path):
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_exclusion_keys(self, path):
        excluded_ids = set()
        excluded_texts = set()
        if path is None:
            return excluded_ids, excluded_texts

        records = self._load_jsonl(path)
        for record in records:
            record_id = self._record_id(record)
            normalized_text = self._normalize_text(record.get("text", ""))
            if record_id is not None:
                excluded_ids.add(record_id)
            if normalized_text:
                excluded_texts.add(normalized_text)
        return excluded_ids, excluded_texts

    def _select_anchor_pool(self, records, excluded_ids, excluded_texts):
        eligible = []
        seen_ids = set()
        seen_texts = set()
        excluded_count = 0
        duplicate_count = 0

        for record in records:
            record_id = self._record_id(record)
            normalized_text = self._normalize_text(record.get("text", ""))
            if (
                (record_id is not None and record_id in excluded_ids)
                or (normalized_text and normalized_text in excluded_texts)
            ):
                excluded_count += 1
                continue
            if not normalized_text:
                continue
            if (
                (record_id is not None and record_id in seen_ids)
                or normalized_text in seen_texts
            ):
                duplicate_count += 1
                continue

            if record_id is not None:
                seen_ids.add(record_id)
            seen_texts.add(normalized_text)
            eligible.append(record)

        return eligible, excluded_count, duplicate_count

    def generate_simulated(
        self,
        jsonl_file_name,
        output_file_name,
        exclude_jsonl=None,
        target_records=600,
        sample_size=4000,
        seed=152,
    ):
        if target_records <= 0:
            raise ValueError("target_records must be positive")
        if sample_size < target_records:
            raise ValueError("sample_size must be at least target_records")

        jsonl_file_path = self._resolve_path(jsonl_file_name)
        output_path = self._resolve_path(output_file_name)
        exclusion_path = (
            self._resolve_path(exclude_jsonl) if exclude_jsonl is not None else None
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._load_jsonl(jsonl_file_path)
        excluded_ids, excluded_texts = self._load_exclusion_keys(exclusion_path)
        eligible, excluded_count, duplicate_count = self._select_anchor_pool(
            data, excluded_ids, excluded_texts
        )

        random_generator = random.Random(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"Input records: {len(data):,}")
        if exclusion_path is not None:
            print(f"Excluded training records: {excluded_count:,}")
            print(f"Excluded IDs: {len(excluded_ids):,}")
            print(f"Excluded anchor texts: {len(excluded_texts):,}")
        print(f"Duplicate source records skipped: {duplicate_count:,}")
        print(f"Eligible anchors: {len(eligible):,}")
        print(f"Requested valid records: {target_records:,}")
        print(f"Selection seed: {seed}")

        if len(eligible) < target_records:
            print(
                f"WARNING: only {len(eligible):,} eligible anchors are available; "
                "the output will contain fewer records."
            )
        if len(eligible) < sample_size:
            print(
                f"WARNING: only {len(eligible):,} eligible anchors are available; "
                f"sampling all of them instead of {sample_size:,}."
            )

        ranks = ["weak", "moderate", "strong"]
        weights = [0.2, 0.3, 0.5]
        sampled = random_generator.sample(eligible, min(sample_size, len(eligible)))
        if len(sampled) < target_records:
            print(
                f"WARNING: sampled only {len(sampled):,} anchors, fewer than the "
                f"requested {target_records:,} valid records."
            )
        temporary_output = output_path.with_name(f".{output_path.name}.tmp")
        with temporary_output.open("w", encoding="utf-8") as out_f:
            valid_count = 0
            for i in range(len(sampled)):
                valid_flag = True
                sampled_record = dict(sampled[i])
                anchor_text = str(sampled_record.get("text", "")).strip()

                if len(anchor_text.split()) > 10:
                    print(f"ORIGINAL: {anchor_text}")
                    
                    rank = random_generator.choices(ranks, weights, k=1)[0]
                    attempt = 0
                    this_type = "inconsistent"
                    while attempt < MAX_ATTEMPTS:
                        prompt = f'For {rank}ly {this_type}, Input: "{anchor_text}"\nOutput:'
                        response = self.chat(prompt, type=this_type)
                        response_json, flag = self.validate_json(response)
                        print(response_json)
                        if (
                            flag == TRUE_FLAG
                            and isinstance(response_json, dict)
                            and isinstance(response_json.get(this_type), str)
                            and response_json[this_type].strip()
                        ):
                            
                            sampled_record.update(response_json)
                            sampled_record[f"{this_type}_rank"] = rank
                            break
                        attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        valid_flag = False
                    rank = random_generator.choices(ranks, weights, k=1)[0]
                    this_type = "paraphrased"
                    attempt = 0
                    while attempt < MAX_ATTEMPTS:
                        prompt = f'For {rank}ly {this_type}, Input: "{anchor_text}"\nOutput:'
                        response = self.chat(prompt, type=this_type)
                        response_json, flag = self.validate_json(response)
                        print(response_json)
                        if (
                            flag == TRUE_FLAG
                            and isinstance(response_json, dict)
                            and isinstance(response_json.get(this_type), str)
                            and response_json[this_type].strip()
                        ):
                            
                            sampled_record.update(response_json)
                            sampled_record[f"{this_type}_rank"] = rank
                            break
                        attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        valid_flag = False

                    rank = random_generator.choices(ranks, weights, k=1)[0]
                    this_type = "randomized"
                    attempt = 0
                    while attempt < MAX_ATTEMPTS:
                        prompt = f'For {rank}ly {this_type}, Input: "{anchor_text}"\nOutput:'
                        response = self.chat(prompt, type=this_type)
                        response_json, flag = self.validate_json(response)
                        print(response_json)
                        if (
                            flag == TRUE_FLAG
                            and isinstance(response_json, dict)
                            and isinstance(response_json.get(this_type), str)
                            and response_json[this_type].strip()
                        ):
                            
                            sampled_record.update(response_json)
                            sampled_record[f"{this_type}_rank"] = rank
                            break
                        attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        valid_flag = False
                    if valid_flag:
                        print(f"Saved {valid_count+1} items")
                        out_f.write(json.dumps(sampled_record) + "\n")
                        valid_count += 1
                    if valid_count >= target_records:
                        break
        temporary_output.replace(output_path)
        print(f"Synthetic output: {output_path}")
        print(f"Valid records written: {valid_count:,}")
        return {
            "input_records": len(data),
            "excluded_records": excluded_count,
            "duplicate_source_records": duplicate_count,
            "eligible_anchors": len(eligible),
            "sampled_anchors": len(sampled),
            "valid_records": valid_count,
            "output": str(output_path),
        }


    def validate_json(self, text:str):
        
        if text == "":
            return "", FALSE_FLAG
        start = text.find("{")
        if start != -1:
            end = text.rfind("}")
            if end != -1:
                if text[start:end+1].count("{") == text[start:end+1].count("}"): #"balanced curly braces"
                    pattern = r"(?<!\w)'([^']*?)'(?!\w)" #To avoid issues with single quotes in texts and keeping apostrophes.
                    text = re.sub(pattern, r'"\1"', text[start:end+1])
                    try:
                        return json.loads(text), TRUE_FLAG
                    except json.JSONDecodeError:
                        return text, MALFORMED_FLAG


        return "", FALSE_FLAG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic O-RAN variants with the trained Llama adapter."
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=ORAN_JSONL,
        help="Anchor JSONL, resolved relative to the project directory if not absolute.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=SYNTHETIC_OUTPUT,
        help="Synthetic output JSONL, resolved relative to the project directory if not absolute.",
    )
    parser.add_argument(
        "--exclude_jsonl",
        type=Path,
        default=None,
        help="Existing synthetic JSONL whose anchor IDs/texts must be excluded.",
    )
    parser.add_argument("--adapter-checkpoint", "--adapter", dest="adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument(
        "--base-model",
        default=MODEL_NAME,
        help="Public model identifier or compatible local base-model checkpoint.",
    )
    parser.add_argument("--examples", type=Path, default=EXAMPLES_PATH)
    parser.add_argument(
        "--target_records",
        "--max_records",
        dest="target_records",
        type=int,
        default=600,
        help="Number of valid synthetic records to write.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=4000,
        help="Number of eligible anchors to consider for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=152,
        help="Seed for anchor selection, variant ranks, and generation sampling.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = LlamaChat._resolve_path(args.input_jsonl)
    output_path = LlamaChat._resolve_path(args.output_jsonl)
    exclusion_path = LlamaChat._resolve_path(args.exclude_jsonl) if args.exclude_jsonl else None
    manifest_path = (
        LlamaChat._resolve_path(args.manifest)
        if args.manifest
        else Path(str(output_path) + ".manifest.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if exclusion_path is not None and not exclusion_path.is_file():
        raise FileNotFoundError(exclusion_path)
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing generation output(s): "
            + ", ".join(map(str, existing))
        )

    chatModel = LlamaChat(args.adapter, args.examples, args.base_model)
    # response = chatModel.chat("In summary, MEC as another key pillar of the convergence of communication and computing, are now separately designed, deployed and operated on top of the mobile network. Thus, the dynamic coordination of the mobile network and application processing is not supported to ensure the end to end service QoS", type="inconsistent")
    # print(response)

    # response = chatModel.chat("In summary, MEC as another key pillar of the convergence of communication and computing, are now separately designed, deployed and operated on top of the mobile network. Thus, the dynamic coordination of the mobile network and application processing is not supported to ensure the end to end service QoS", type="paraphrase")
    # print(response)

    # response = chatModel.chat("In summary, MEC as another key pillar of the convergence of communication and computing, are now separately designed, deployed and operated on top of the mobile network. Thus, the dynamic coordination of the mobile network and application processing is not supported to ensure the end to end service QoS", type="randomize")
    # print(response)
    
    generation = chatModel.generate_simulated(
        args.input_jsonl,
        args.output_jsonl,
        exclude_jsonl=args.exclude_jsonl,
        target_records=args.target_records,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "exclude_jsonl": str(exclusion_path) if exclusion_path else None,
        "exclude_sha256": sha256_file(exclusion_path) if exclusion_path else None,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "adapter": str(chatModel.adapter_path),
        "adapter_sha256": sha256_tree(chatModel.adapter_path),
        "base_model": MODEL_NAME,
        "base_model_revision": MODEL_REVISION,
        "examples": str(chatModel.examples_path),
        "examples_sha256": sha256_file(chatModel.examples_path),
        "target_records": args.target_records,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "max_attempts_per_variant": MAX_ATTEMPTS,
        "generation": generation,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Generation manifest: {manifest_path}")
