import os
import sys
import cv2
import torch
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import draw_bounding_boxes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("[WARN] Установите torchmetrics: pip install torchmetrics")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.VAE import VAE

def calculate_mse_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0: return 0, 100
    return mse, 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_fid_and_is(real_dir, fake_paths, device):
    if not HAS_METRICS or not fake_paths: return -1.0, -1.0
    fid_metric = FrechetInceptionDistance(feature=64).to(device)
    is_metric = InceptionScore().to(device)
    
    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    if not real_paths: return -1.0, -1.0
    
    real_tensors = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in real_paths]
    fake_tensors = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in fake_paths]
    
    fid_metric.update(torch.stack(real_tensors).to(device), real=True)
    fake_stack = torch.stack(fake_tensors).to(device)
    fid_metric.update(fake_stack, real=False)
    is_metric.update(fake_stack)
    
    is_mean, _ = is_metric.compute()
    return fid_metric.compute().item(), is_mean.item()

def create_sample_grid_with_bboxes(aug_img_dir, aug_lbl_dir, save_path):
    images = [f for f in os.listdir(aug_img_dir) if f.endswith(('.jpg', '.png'))][:6]
    if not images: return
    
    drawn_tensors = []
    for img_name in images:
        img_path = os.path.join(aug_img_dir, img_name)
        lbl_path = os.path.join(aug_lbl_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        img_tensor = torch.from_numpy(np.array(Image.open(img_path).convert("RGB"))).permute(2, 0, 1)
        bboxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        H, W = img_tensor.shape[1], img_tensor.shape[2]
                        x1, y1 = int((xc - w/2) * W), int((yc - h/2) * H)
                        x2, y2 = int((xc + w/2) * W), int((yc + h/2) * H)
                        bboxes.append([x1, y1, x2, y2])
        if bboxes:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float)
            img_tensor = draw_bounding_boxes(img_tensor, boxes_tensor, colors="red", width=2)
        drawn_tensors.append(img_tensor)
        
    grid = vutils.make_grid(drawn_tensors, nrow=3, padding=5, normalize=False)
    vutils.save_image(grid.float() / 255.0, save_path)

def plot_beautiful_metrics(psnr_values, fid_val, is_val, save_dir, arch_name):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Оценка качества генерации: {arch_name}', fontsize=16, fontweight='bold')
    
    if psnr_values:
        sns.histplot(psnr_values, kde=True, ax=axes[0], color="skyblue")
        axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', label=f'Mean: {np.mean(psnr_values):.2f}')
        axes[0].set_title('Распределение PSNR (Top-K)', fontsize=12)
        axes[0].legend()
        
    sns.barplot(x=[arch_name], y=[fid_val], ax=axes[1], palette="flare", hue=[arch_name], legend=False)
    axes[1].set_title('Frechet Inception Distance (↓)', fontsize=12)
    axes[1].bar_label(axes[1].containers[0], fmt='%.2f')
    
    sns.barplot(x=[arch_name], y=[is_val], ax=axes[2], palette="crest", hue=[arch_name], legend=False)
    axes[2].set_title('Inception Score (↑)', fontsize=12)
    axes[2].bar_label(axes[2].containers[0], fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_evaluation.png'), dpi=300)
    plt.close()

def run_scientific_augmentation_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH_NAME = "VAE"
    print(f"[INIT] Запуск генерации {ARCH_NAME} на устройстве: {device}")

    IMG_SIZE, NUM_CLASSES, LATENT_DIM, AUG_FACTOR = 128, 6, 256, 5
    TOP_K_LIMITS = {"train": 105, "val": 25, "test": 0}
    
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "vae_latest.pth")
    BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_final_128")
    AUG_DIR = os.path.join(PROJECT_ROOT, "datasets", ARCH_NAME)
    SPLITS = ['train', 'val', 'test']

    model = VAE(in_channels=3, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    all_psnr_scores = []

    for split in SPLITS:
        limit_k = TOP_K_LIMITS.get(split, 0)
        orig_img_dir = os.path.join(BASE_DATA_DIR, split, "images")
        orig_lbl_dir = os.path.join(BASE_DATA_DIR, split, "labels")
        aug_img_dir = os.path.join(AUG_DIR, split, "images")
        aug_lbl_dir = os.path.join(AUG_DIR, split, "labels")
        temp_dir = os.path.join(AUG_DIR, split, "temp")

        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_lbl_dir, exist_ok=True)

        if not os.path.exists(orig_img_dir): continue
        if limit_k > 0: os.makedirs(temp_dir, exist_ok=True)

        all_images = [f for f in os.listdir(orig_img_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"\nСплит [{split.upper()}]. Оригиналов: {len(all_images)}")

        class_candidates = {c: [] for c in range(NUM_CLASSES)}

        for img_name in tqdm(all_images, desc=f"Генерация {split}"):
            base_name = os.path.splitext(img_name)[0]
            src_img_path = os.path.join(orig_img_dir, img_name)
            src_lbl_path = os.path.join(orig_lbl_dir, base_name + ".txt")
            
            if not os.path.exists(src_lbl_path): continue
            with open(src_lbl_path, "r") as f: original_lines = f.readlines()
            
            shutil.copy(src_img_path, os.path.join(aug_img_dir, img_name))
            shutil.copy(src_lbl_path, os.path.join(aug_lbl_dir, base_name + ".txt"))
            
            if limit_k == 0 or not original_lines: continue

            try:
                bg_pil = Image.open(src_img_path).convert("RGB")
                bg_tensor = transform(bg_pil).unsqueeze(0).to(device)
                orig_cv = cv2.resize(cv2.imread(src_img_path), (IMG_SIZE, IMG_SIZE))
            except: continue

            for aug_idx in range(AUG_FACTOR):
                new_yolo_lines = []
                cond_mask = np.zeros((1, NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                current_target_class = None
                
                for line in original_lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    orig_class = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    target_class = random.randint(0, NUM_CLASSES - 1)
                    if current_target_class is None: current_target_class = target_class
                    
                    new_yolo_lines.append(f"{target_class} {xc} {yc} {w} {h}\n")
                    x_min, y_min = max(0, int((xc - w / 2) * IMG_SIZE)), max(0, int((yc - h / 2) * IMG_SIZE))
                    x_max, y_max = min(IMG_SIZE, int((xc + w / 2) * IMG_SIZE)), min(IMG_SIZE, int((yc + h / 2) * IMG_SIZE))
                    cond_mask[0, target_class, y_min:y_max, x_min:x_max] = 1.0

                for c in range(NUM_CLASSES):
                    if cond_mask[0, c].sum() > 0:
                        cond_mask[0, c] = cv2.GaussianBlur(cond_mask[0, c], (11, 11), 3)

                cond_tensor = torch.tensor(cond_mask).to(device)
                mask_sum = torch.clamp(cond_tensor.sum(dim=1, keepdim=True), 0, 1)
                z_random = torch.randn(1, LATENT_DIM).to(device)
                
                with torch.no_grad():
                    generated_tensor = model.decode(z_random, cond_tensor)
                    x_fake = (bg_tensor * (1 - mask_sum)) + (generated_tensor * mask_sum)

                img_np = np.clip(x_fake[0].cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                clean_aug_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                _, psnr_val = calculate_mse_psnr(orig_cv, clean_aug_bgr)
                
                temp_img = os.path.join(temp_dir, f"temp_{base_name}_{aug_idx}.jpg")
                temp_lbl = os.path.join(temp_dir, f"temp_{base_name}_{aug_idx}.txt")
                cv2.imwrite(temp_img, clean_aug_bgr)
                with open(temp_lbl, "w") as f_out: f_out.writelines(new_yolo_lines)
                
                class_candidates[current_target_class].append({
                    "img": temp_img, "lbl": temp_lbl, "score": psnr_val,
                    "fin_img": f"aug_{aug_idx}_{img_name}", "fin_lbl": f"aug_{aug_idx}_{base_name}.txt"
                })

        if limit_k > 0:
            print(f"\n[FILTER] Отбор Top-{limit_k} кандидатов...")
            selected_paths = []
            for c in range(NUM_CLASSES):
                class_candidates[c].sort(key=lambda x: x["score"], reverse=True)
                for item in class_candidates[c][:limit_k]:
                    shutil.move(item["img"], os.path.join(aug_img_dir, item["fin_img"]))
                    shutil.move(item["lbl"], os.path.join(aug_lbl_dir, item["fin_lbl"]))
                    selected_paths.append(os.path.join(aug_img_dir, item["fin_img"]))
                    all_psnr_scores.append(item["score"])
                    
            shutil.rmtree(temp_dir, ignore_errors=True)
            fid_val, is_val = calculate_fid_and_is(orig_img_dir, selected_paths, device)
            print(f"[METRICS] Сплит {split} | FID: {fid_val:.2f} | IS: {is_val:.2f}")
            
            # Сохраняем пример сетки с боксами
            create_sample_grid_with_bboxes(aug_img_dir, aug_lbl_dir, os.path.join(AUG_DIR, f"sample_grid_{split}.png"))

    # Отрисовка графиков в папке архитектуры
    plot_beautiful_metrics(all_psnr_scores, fid_val, is_val, os.path.join(AUG_DIR, "plots"), ARCH_NAME)
    print("\n[DONE] Генерация и оценка завершены.")

if __name__ == "__main__":
    run_scientific_augmentation_pipeline()