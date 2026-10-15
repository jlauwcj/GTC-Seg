import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
from tqdm import tqdm
import sys

# ================= 配置区域 =================

# 1. 本地模型权重路径
LOCAL_MODEL_PATH = Path("./checkpoints/clipseg-rd64-refined")

# 2. SemanticKITTI 数据集根目录
DATA_ROOT = Path("/media/dell/wcjdata/nzz/PointCT-main/dataset/SemanticKITTI/sequences/")

# 3. 输出目录
OUTPUT_ROOT = Path("/media/dell/wcjdata/nzz/PointCT-main/dataset/MaskCLIP_Results/sequences")

# 4. 指定要处理的序列
SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# 5. 类别文本提示
CLASS_PROMPTS = [
    "car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other ground", "building", "fence",
    "vegetation", "trunk", "terrain", "pole", "traffic sign"
]
# ===========================================

def main():
    if not LOCAL_MODEL_PATH.exists():
        print(f"【错误】找不到本地模型目录: {LOCAL_MODEL_PATH}")
        sys.exit(1)

    print("正在从本地加载 CLIPSeg 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = CLIPSegProcessor.from_pretrained(str(LOCAL_MODEL_PATH))
        model = CLIPSegForImageSegmentation.from_pretrained(str(LOCAL_MODEL_PATH)).to(device)
        
        # ==========================================================
        # 【核心修复】修改 Processor 内部配置，强制将图片调整为 224x224
        # 这样就不需要在调用时传参，避免了报错
        # ==========================================================
        if hasattr(processor, "image_processor"):
            processor.image_processor.size = {"height": 224, "width": 224}
            processor.image_processor.crop_size = {"height": 224, "width": 224}
            print(">>> 已强制修改 Processor 输入尺寸配置为 224x224")

    except Exception as e:
        print(f"【加载失败】: {e}")
        sys.exit(1)

    model.eval()

    for seq in SEQUENCES:
        img_dir = DATA_ROOT / seq / "image_2"
        out_dir = OUTPUT_ROOT / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(list(img_dir.glob("*.png")))
        if len(images) == 0:
            continue
            
        print(f"Processing Sequence {seq} ({len(images)} frames)...")

        for img_path in tqdm(images):
            save_path = out_dir / f"{img_path.stem}.npy"
            if save_path.exists():
                continue

            try:
                # 1. 加载图片
                image = Image.open(img_path).convert("RGB")
                original_size = image.size # (W, H)

                # 2. 准备输入
                # 【注意】这里不再传 size 参数，直接使用上面修改过的配置
                inputs = processor(
                    text=CLASS_PROMPTS, 
                    images=[image] * len(CLASS_PROMPTS), 
                    padding="max_length", 
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 3. 推理
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # outputs.logits shape: [Batch, H_feat, W_feat]
                logits = outputs.logits
                
                # 4. 后处理：上采样回原图尺寸
                logits = logits.unsqueeze(0) 
                
                # 插值回原始图片分辨率
                logits = torch.nn.functional.interpolate(
                    logits, 
                    size=(original_size[1], original_size[0]), # (H, W)
                    mode="bilinear", 
                    align_corners=False
                )
                
                # 5. 获取预测类别
                preds = torch.argmax(logits[0], dim=0) 
                
                # 6. 保存
                np.save(save_path, preds.cpu().numpy().astype(np.uint8))
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("\n所有 MaskCLIP 伪标签生成完毕！")

if __name__ == "__main__":
    main()
