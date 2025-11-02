import os
import json
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception as e: 
    raise RuntimeError("PEFT is required for LoRA training. Please install `peft`.\n" + str(e))

warnings.filterwarnings("ignore")



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MSRA_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "msra"))

DEFAULT_MODEL_NAME = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(THIS_DIR, "qwen-ner-lora"))

TAG_TYPES: List[str] = [
    "O",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
]
ID2LABEL = {i: label for i, label in enumerate(TAG_TYPES)}
LABEL2ID = {label: i for i, label in enumerate(TAG_TYPES)}



def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_msra_split(split_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load MSRA split from `split_dir` containing `sentences.txt` and `tags.txt`.
    Returns a list of dicts: {"text": str, "tags": List[str]} where tags are spaced strings per char.
    """
    sentences_path = os.path.join(split_dir, "sentences.txt")
    tags_path = os.path.join(split_dir, "tags.txt")

    texts = _read_lines(sentences_path)
    texts = [t.replace(" ", "").strip() for t in texts]

    tags_lines = _read_lines(tags_path)
    tags = [line.strip().split(" ") for line in tags_lines]

    if max_samples is not None:
        texts = texts[:max_samples]
        tags = tags[:max_samples]

    data = []
    for text, tag_seq in zip(texts, tags):
        data.append({"text": text, "tags": tag_seq})
    return data


def tags_to_entities(text: str, tags: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO tags (per character) to entity list: [{"text": str, "type": str}].
    """
    entities: List[Dict[str, Any]] = []
    current_entity = []
    current_type: Optional[str] = None

    for ch, tag in zip(text, tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append({"text": "".join(current_entity), "type": current_type})
            current_entity = [ch]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_entity and current_type == tag[2:]:
            current_entity.append(ch)
        else:
            if current_entity:
                entities.append({"text": "".join(current_entity), "type": current_type})
            current_entity = []
            current_type = None

    if current_entity:
        entities.append({"text": "".join(current_entity), "type": current_type})

    return entities


def build_instruction_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert MSRA samples into instruction-tuning pairs for chat models.
    Each item: {"prompt": str, "response": str}
    """
    system_prompt = (
        "你是一个命名实体识别助手。给定一段中文文本，识别其中的人名(PER)、"
        "地名(LOC)、组织名(ORG)等实体，并返回一个JSON数组，数组中的每一项"
        "包含{text, type}字段，其中type∈{PER, LOC, ORG}。不包含其他内容。"
    )

    result: List[Dict[str, str]] = []
    for item in samples:
        text: str = item["text"]
        tags: List[str] = item["tags"]
        entities = tags_to_entities(text, tags)
        entities = [e for e in entities if e.get("type") in {"PER", "LOC", "ORG"}]
        target_json = json.dumps(entities, ensure_ascii=False)

        prompt = (
            f"系统: {system_prompt}\n"
            f"用户: 文本如下：\n{text}\n\n请只输出JSON数组，不要解释。"
        )
        response = target_json
        result.append({"prompt": prompt, "response": response})
    return result



@dataclass
class SupervisedNERCollator:
    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        for sample in batch:
            prompt = sample["prompt"]
            response = sample["response"]

            use_chat = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None

            if use_chat:
                system_msg = {"role": "system", "content": ""}
                prompt_ids = self.tokenizer.apply_chat_template(
                    [
                        system_msg,
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,  # include assistant prefix
                    return_tensors="pt",
                )[0]

                full_ids = self.tokenizer.apply_chat_template(
                    [
                        system_msg,
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )[0]
            else:
                prompt_rendered = f"<s>[SYSTEM]\n{sample['prompt']}\n</s>\n[USER]\n{sample['prompt']}\n</s>\n[ASSISTANT]"
                full_rendered = f"{prompt_rendered} {sample['response']}\n</s>"

                prompt_ids = self.tokenizer(
                    prompt_rendered,
                    add_special_tokens=True,
                    return_tensors="pt",
                )["input_ids"][0]
                full_ids = self.tokenizer(
                    full_rendered,
                    add_special_tokens=True,
                    return_tensors="pt",
                )["input_ids"][0]

            if full_ids.size(0) > self.max_length:
                full_ids = full_ids[-self.max_length:]
                prompt_len = min(prompt_ids.size(0), full_ids.size(0) - 1)
            else:
                prompt_len = min(prompt_ids.size(0), full_ids.size(0) - 1)

            labels = full_ids.clone()
            labels[:prompt_len] = -100 
            attention_mask = torch.ones_like(full_ids)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }



def maybe_enable_4bit_kwargs() -> Dict[str, Any]:
    """Return kwargs for 4-bit loading if bitsandbytes is available; otherwise empty."""
    try:
        import bitsandbytes as bnb  
        return {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            "device_map": "auto",
        }
    except Exception:
        return {"device_map": "auto"}


def init_model_and_tokenizer(model_name: str) -> (AutoModelForCausalLM, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **maybe_enable_4bit_kwargs(),
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tok



def build_hf_dataset(train_pairs: List[Dict[str, str]], eval_pairs: List[Dict[str, str]]) -> (Dataset, Dataset):
    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)
    return train_ds, eval_ds


def train(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    train_max_samples: int = 2000,
    eval_max_samples: int = 200,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    seed: int = 42,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading MSRA dataset from: {MSRA_DIR}")
    train_raw = load_msra_split(os.path.join(MSRA_DIR, "train"), max_samples=train_max_samples)
    eval_raw = load_msra_split(os.path.join(MSRA_DIR, "val"), max_samples=eval_max_samples)

    train_pairs = build_instruction_samples(train_raw)
    eval_pairs = build_instruction_samples(eval_raw)

    model, tokenizer = init_model_and_tokenizer(model_name)
    collator = SupervisedNERCollator(tokenizer=tokenizer, max_length=max_length)
    train_ds, eval_ds = build_hf_dataset(train_pairs, eval_pairs)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Starting LoRA training on Qwen...")
    trainer.train()

    # Save LoRA adapters only
    print(f"Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



@torch.inference_mode()
def ner_infer(
    text: str,
    base_model_name: str,
    lora_dir: str,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **maybe_enable_4bit_kwargs(),
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    system_prompt = (
        "你是一个命名实体识别助手。给定一段中文文本，识别其中的人名(PER)、"
        "地名(LOC)、组织名(ORG)等实体，并返回一个JSON数组，数组中的每一项"
        "包含{text, type}字段，其中type∈{PER, LOC, ORG}。不包含其他内容。"
    )
    prompt = (
        f"系统: {system_prompt}\n"
        f"用户: 文本如下：\n{text}\n\n请只输出JSON数组，不要解释。"
    )

    use_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None
    if use_chat:
        conv = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        input_ids = tok.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        rendered = f"[USER]\n{prompt}\n[/USER]\n[ASSISTANT]"
        input_ids = tok(rendered, return_tensors="pt").to(model.device)["input_ids"]

    gen = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    output_text = tok.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)

    json_start = output_text.find("[")
    json_end = output_text.find("]", json_start)
    if json_start != -1 and json_end != -1:
        snippet = output_text[json_start: json_end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def demo_after_training():
    """Run a tiny demo on a few sentences after training."""
    examples = [
        "今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。",
        "人工智能是未来的希望，也是中国和美国的冲突点。",
        "明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。",
    ]
    for text in examples:
        ents = ner_infer(text, DEFAULT_MODEL_NAME, OUTPUT_DIR)
        print(f"文本: {text}")
        if ents:
            for ent in ents:
                print(f"  {ent.get('type')}: {ent.get('text')}")
        else:
            print("  未识别到实体")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen LoRA 微调中文NER (MSRA)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--train_max_samples", type=int, default=2000)
    parser.add_argument("--eval_max_samples", type=int, default=200)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="训练完成后跑一个简短推理demo")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_max_samples=args.train_max_samples,
        eval_max_samples=args.eval_max_samples,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )

    if args.demo:
        demo_after_training()


