import os
import shutil
from ultralytics import YOLO

# KONFIGURASI PATH
MODEL_PATH = r"C:\Workspace\projects\ml\real-world-telecom-infrastructure-object-detection\deployment\best.pt"
DATA_YAML = r"C:\Workspace\projects\ml\real-world-telecom-infrastructure-object-detection\data\processed\data.yaml"

# KONFIGURASI VALIDASI
IMG_SIZE = 1248
PROJECT_DIR = r"C:\Workspace\projects\ml\real-world-telecom-infrastructure-object-detection\experiments"
EXPERIMENT_NAME = "validation_report"


def main():
    print("=== MEMULAI VALIDASI MODEL (YOLOv8) ===")

    # 1. Cek Ketersediaan File
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan di: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] Data Config tidak ditemukan di: {DATA_YAML}")
        return

    print(f"Model  : {os.path.basename(MODEL_PATH)}")
    print(f"Data   : {os.path.basename(DATA_YAML)}")
    print(f"Output : {os.path.join(PROJECT_DIR, EXPERIMENT_NAME)}")
    print("-" * 40)

    # 2. Load Model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    # 3. Jalankan Validasi
    # split='val' -> Menggunakan data validasi yang ada di data.yaml
    try:
        metrics = model.val(
            data=DATA_YAML,
            imgsz=IMG_SIZE,
            batch=4,  # Sesuaikan dengan VRAM lokal (4/8 aman untuk 1248px)
            conf=0.25,  # Confidence Threshold
            iou=0.6,  # NMS IoU Threshold
            split='val',
            project=PROJECT_DIR,
            name=EXPERIMENT_NAME,
            exist_ok=True,  # Timpa jika folder sudah ada
            plots=True,  # WAJIB TRUE: Untuk generate Confusion Matrix & Kurva
            device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'  # Auto deteksi GPU/CPU
        )

        print("\n[SUKSES] Validasi Selesai!")

        # 4. Lokasi File Confusion Matrix
        result_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME)
        cm_file = os.path.join(result_dir, 'confusion_matrix.png')

        if os.path.exists(cm_file):
            print(f"[INFO] Confusion Matrix tersimpan di:\n -> {cm_file}")
            # Opsional: Buka folder otomatis (Windows Only)
            os.startfile(result_dir)
        else:
            print("[WARNING] Gambar Confusion Matrix tidak ditemukan. Cek folder output manual.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Terjadi kesalahan saat validasi:\n{e}")
        print("\nTips Troubleshooting:")
        print("1. Cek isi 'data.yaml', pastikan path 'val:' mengarah ke folder gambar yang benar.")
        print("2. Jika VRAM kurang, kurangi 'batch=4' menjadi 'batch=1'.")


if __name__ == "__main__":
    main()