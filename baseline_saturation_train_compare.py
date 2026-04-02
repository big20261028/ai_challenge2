# ============================================================
#  baseline - 채도별 학습 비교 실험 (PC 3대 분산 실행용)
# ============================================================
#
#  실행 방법
#  ---------
#  1. 아래 [채도 설정] 블록에서 사용할 PC에 맞는 TRAIN_DIR 한 줄만 주석 해제
#  2. 코랩에서 실행
#
#  PC별 역할
#  ----------
#  PC-A : TRAIN_DIR = ".../train"                ← 원본 채도 (1.0)
#  PC-B : TRAIN_DIR = ".../train_saturation_1.5" ← 채도 1.5배
#  PC-C : TRAIN_DIR = ".../train_saturation_2.5" ← 채도 2.5배
#
# ============================================================


# ─────────────────────────────────────────────────────────────
# 환경 설치
# ─────────────────────────────────────────────────────────────
# !pip -q install git+https://github.com/huggingface/transformers accelerate
# !pip -q install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# !pip -q install "peft>=0.13.2" "bitsandbytes==0.46.1" datasets pillow pandas --upgrade


# ─────────────────────────────────────────────────────────────
# 구글 드라이브 마운트 & 압축 해제
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# 압축 파일 이름이 다르면 아래 경로를 수정하세요
!unzip -q "/content/drive/My Drive/2026-ssafy-15-2-ai.zip" -d "/content/"


# ─────────────────────────────────────────────────────────────
# 라이브러리
# ─────────────────────────────────────────────────────────────
import os, math, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
from typing import Any
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ─────────────────────────────────────────────────────────────
# ★ [채도 설정] 이 PC에서 사용할 줄 하나만 주석 해제하세요 ★
# ─────────────────────────────────────────────────────────────

TRAIN_DIR = "/content/train"                  # PC-A : 원본 채도 (1.0)
# TRAIN_DIR = "/content/train_saturation_1.5" # PC-B : 채도 1.5배
# TRAIN_DIR = "/content/train_saturation_2.5" # PC-C : 채도 2.5배

# ─────────────────────────────────────────────────────────────

# 설정된 채도를 제출 파일명과 로그에 자동 반영
if   "2.5" in TRAIN_DIR: SAT_TAG = "sat2.5"
elif "1.5" in TRAIN_DIR: SAT_TAG = "sat1.5"
else:                     SAT_TAG = "sat1.0"

print(f"학습 이미지 폴더 : {TRAIN_DIR}")
print(f"채도 태그        : {SAT_TAG}")


# ─────────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_SIZE = 384
SEED       = 42
GRAD_ACCUM = 4

BASE_DIR = "/content"
TEST_DIR = f"{BASE_DIR}/test"                  # 테스트 이미지는 항상 원본
SAVE_DIR = f"{BASE_DIR}/model_{SAT_TAG}"       # 모델 저장 위치

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
test_df  = pd.read_csv(f"{BASE_DIR}/test.csv")

# CSV의 path 컬럼(예: "train/train_0001.jpg")에서 파일명만 추출해
# 선택한 TRAIN_DIR 경로로 교체
train_df["path"] = train_df["path"].apply(
    lambda p: os.path.join(TRAIN_DIR, os.path.basename(p))
)
# 테스트는 항상 원본 폴더
test_df["path"] = test_df["path"].apply(
    lambda p: os.path.join(TEST_DIR, os.path.basename(p))
)

train_df = train_df.sample(n=200, random_state=SEED).reset_index(drop=True)

split        = int(len(train_df) * 0.9)
train_subset = train_df.iloc[:split].reset_index(drop=True)
valid_subset = train_df.iloc[split:].reset_index(drop=True)

print(f"학습: {len(train_subset)}개 / 검증: {len(valid_subset)}개 / 테스트: {len(test_df)}개")


# ─────────────────────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────────────────────
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


# ─────────────────────────────────────────────────────────────
# Dataset & Collator
# ─────────────────────────────────────────────────────────────
class VQAMCDataset(Dataset):
    def __init__(self, df, processor, train=True):
        self.df        = df.reset_index(drop=True)
        self.processor = processor
        self.train     = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")

        user_text = build_mc_prompt(
            str(row["question"]),
            str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user",   "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  user_text},
            ]},
        ]
        if self.train:
            gold = str(row["answer"]).strip().lower()
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": gold}]})

        return {"messages": messages, "image": img}


@dataclass
class DataCollator:
    processor: Any
    train: bool = True

    def __call__(self, batch):
        texts, images = [], []
        for sample in batch:
            texts.append(self.processor.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            ))
            images.append(sample["image"])

        enc = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )
        if self.train:
            enc["labels"] = enc["input_ids"].clone()
        return enc


# ─────────────────────────────────────────────────────────────
# 모델 & Processor 로드
# ─────────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    trust_remote_code=True,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model = prepare_model_for_kbit_training(base_model)
base_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ─────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────
train_ds     = VQAMCDataset(train_subset, processor, train=True)
valid_ds     = VQAMCDataset(valid_subset, processor, train=True)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=DataCollator(processor, True), num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                          collate_fn=DataCollator(processor, True), num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_steps = math.ceil(len(train_loader) / GRAD_ACCUM)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(num_steps * 0.03), num_steps
)
scaler = torch.cuda.amp.GradScaler(enabled=True)

model.train()
for epoch in range(1):
    running = 0.0
    pbar = tqdm(train_loader, desc=f"[{SAT_TAG}] Epoch {epoch+1} train", unit="batch")
    for step, batch in enumerate(pbar, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(**batch).loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        running += loss.item()

        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pbar.set_postfix({"loss": f"{running/GRAD_ACCUM:.3f}"})
            running = 0.0

    # 검증 손실
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for vb in tqdm(valid_loader, desc=f"[{SAT_TAG}] Epoch {epoch+1} valid"):
            vb = {k: v.to(device) for k, v in vb.items()}
            val_loss += model(**vb).loss.item()
            val_steps += 1
    print(f"[{SAT_TAG}] Epoch {epoch+1} | valid loss: {val_loss/val_steps:.4f}")
    model.train()

# 모델 저장
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"✅ 모델 저장: {SAVE_DIR}")


# ─────────────────────────────────────────────────────────────
# 추론  (테스트셋 — 항상 원본 채도 이미지 사용)
# ─────────────────────────────────────────────────────────────
def extract_choice(text: str) -> str:
    text  = text.strip().lower()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: return "a"
    last = lines[-1]
    if last in ["a","b","c","d"]: return last
    for tok in last.split():
        if tok in ["a","b","c","d"]: return tok
    return "a"

model.eval()
preds = []

for i in tqdm(range(len(test_df)), desc=f"[{SAT_TAG}] Inference"):
    row = test_df.iloc[i]
    img = Image.open(row["path"]).convert("RGB")   # 원본 채도 그대로

    user_text = build_mc_prompt(
        row["question"], row["a"], row["b"], row["c"], row["d"]
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user",   "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  user_text},
        ]},
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=2, do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    preds.append(extract_choice(
        processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    ))

# 제출 파일 저장 — 파일명에 채도 태그 자동 포함
submission_path = f"{BASE_DIR}/submission_{SAT_TAG}.csv"
pd.DataFrame({"id": test_df["id"], "answer": preds}).to_csv(submission_path, index=False)
print(f"💾 제출 파일 저장: {submission_path}")
