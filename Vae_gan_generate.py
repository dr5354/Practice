import os
import sys
import shutil
import torch
import random
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import draw_bounding_boxes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.GAN import GeneratorSPADE
from models.VAE_encoder import VAE_Encoder
from data.dataset import DefectDataset

def calculate_psnr(img_real, img_fake):
    real_np = img_real.cpu().numpy().transpose(1, 2, 0)
    fake_np = img_fake.detach().cpu().numpy().transpose(1, 2, 0)
    return psnr(real_np, fake_np, data_range=1.0)

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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metrics_evaluation.png'), dpi=300)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH_NAME = "VAE_GAN"
    print(f"[INIT] Запуск пайплайна {ARCH_NAME} на устройстве: {device}")

    NUM_CLASSES, LATENT_DIM, IMG_SIZE, AUG_MULTIPLIER = 6, 256, 128, 5
    TOP_K_LIMITS = {"train": 105, "val": 25, "test": 0}
    SPLITS = ["train", "val", "test"]

    ORIGINAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_final_128")
    AUGMENTED_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", ARCH_NAME)

    G_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "vaegan_G_epoch_200.pth"
    )
    E_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "vaegan_E_epoch_200.pth"
    )

    netG = GeneratorSPADE(3, NUM_CLASSES, LATENT_DIM).to(device)
    netE = VAE_Encoder(3, LATENT_DIM).to(device)
    netG.load_state_dict(torch.load(G_WEIGHTS, map_location=device))
    netE.load_state_dict(torch.load(E_WEIGHTS, map_location=device))
    netG.eval(); netE.eval()

    all_saved_psnr = []

    for split in SPLITS:
        limit_k = TOP_K_LIMITS.get(split, 0)
        orig_img_dir = os.path.join(ORIGINAL_DATA_DIR, split, "images")
        orig_lbl_dir = os.path.join(ORIGINAL_DATA_DIR, split, "labels")
        aug_img_dir = os.path.join(AUGMENTED_DATA_DIR, split, "images")
        aug_lbl_dir = os.path.join(AUGMENTED_DATA_DIR, split, "labels")
        temp_dir = os.path.join(AUGMENTED_DATA_DIR, split, "temp")

        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_lbl_dir, exist_ok=True)

        if not os.path.exists(orig_img_dir): continue

        for f in os.listdir(orig_img_dir): shutil.copy2(os.path.join(orig_img_dir, f), os.path.join(aug_img_dir, f))
        for f in os.listdir(orig_lbl_dir): shutil.copy2(os.path.join(orig_lbl_dir, f), os.path.join(aug_lbl_dir, f))

        if limit_k == 0: continue
        os.makedirs(temp_dir, exist_ok=True)

        dataset = DefectDataset(img_dir=orig_img_dir, label_dir=orig_lbl_dir, img_size=IMG_SIZE)
        class_candidates = {c: [] for c in range(NUM_CLASSES)}
        
        for idx in tqdm(range(len(dataset)), desc=f"Аугментация [{split}]"):
            orig_img_tensor, _ = dataset[idx]
            orig_img_tensor = orig_img_tensor.to(device)
            img_name = dataset.files[idx]
            base_name = img_name.rsplit(".", 1)[0]
            
            img_for_encoder = (orig_img_tensor * 2.0) - 1.0 
            with torch.no_grad(): mu_style, _ = netE(img_for_encoder.unsqueeze(0))

            for gen_idx in range(AUG_MULTIPLIER):
                num_new_defects = random.randint(1, 3)
                new_bboxes = []
                for _ in range(num_new_defects):
                    w, h = random.uniform(0.05, 0.25), random.uniform(0.05, 0.25)
                    new_bboxes.append((random.randint(0, NUM_CLASSES - 1), random.uniform(w/2, 1.0 - w/2), random.uniform(h/2, 1.0 - h/2), w, h))
                primary_class = new_bboxes[0][0] 
                
                mask = torch.zeros((1, NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=torch.float32).to(device)
                for cls_id, xc, yc, w, h in new_bboxes:
                    x1, y1 = int(max(0, (xc - w / 2) * IMG_SIZE)), int(max(0, (yc - h / 2) * IMG_SIZE))
                    x2, y2 = int(min(IMG_SIZE, (xc + w / 2) * IMG_SIZE)), int(min(IMG_SIZE, (yc + h / 2) * IMG_SIZE))
                    mask[0, cls_id, y1:y2, x1:x2] = 1.0

                with torch.no_grad():
                    fake_img = netG(mu_style, mask)
                    fake_img = torch.clamp((fake_img[0] + 1.0) / 2.0, 0.0, 1.0)

                val_psnr = calculate_psnr(orig_img_tensor, fake_img)
                
                temp_img = os.path.join(temp_dir, f"temp_{idx}_{gen_idx}.jpg")
                temp_lbl = os.path.join(temp_dir, f"temp_{idx}_{gen_idx}.txt")
                vutils.save_image(fake_img, temp_img, normalize=False)
                with open(temp_lbl, "w") as f:
                    for c_id, xc, yc, bw, bh in new_bboxes: f.write(f"{c_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        
                class_candidates[primary_class].append({
                    "img": temp_img, "lbl": temp_lbl, "score": val_psnr,
                    "fin_img": f"aug_{primary_class}_{base_name}_{gen_idx}.jpg", "fin_lbl": f"aug_{primary_class}_{base_name}_{gen_idx}.txt"
                })

        selected_paths = []
        for c in range(NUM_CLASSES):
            class_candidates[c].sort(key=lambda x: x["score"], reverse=True)
            for item in class_candidates[c][:limit_k]:
                shutil.move(item["img"], os.path.join(aug_img_dir, item["fin_img"]))
                shutil.move(item["lbl"], os.path.join(aug_lbl_dir, item["fin_lbl"]))
                selected_paths.append(os.path.join(aug_img_dir, item["fin_img"]))
                all_saved_psnr.append(item["score"])

        shutil.rmtree(temp_dir, ignore_errors=True)
        fid_score, is_score = calculate_fid_and_is(orig_img_dir, selected_paths, device)
        print(f"[METRICS] Сплит {split} | FID: {fid_score:.2f} | IS: {is_score:.2f}")
        create_sample_grid_with_bboxes(aug_img_dir, aug_lbl_dir, os.path.join(AUGMENTED_DATA_DIR, f"sample_grid_{split}.png"))

    plot_beautiful_metrics(all_saved_psnr, fid_score, is_score, os.path.join(AUGMENTED_DATA_DIR, "plots"), ARCH_NAME)
    print("\n[DONE] Полный пайплайн завершен!")

if __name__ == "__main__":
    main()