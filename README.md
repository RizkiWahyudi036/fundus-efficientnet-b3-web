# Klasifikasi Gangguan Mata Berbasis Citra Fundus (EfficientNet-B3)

Prototipe antarmuka web penelitian skripsi Program Studi S-1 Ilmu Komputer, Fakultas Ilmu Komputer dan Teknologi Informasi, Universitas Sumatera Utara (Rizki Wahyudi, 211401036).

Aplikasi ini memuat enam model ONNX (`opset` 17) hasil eksperimen terkontrol tiga skenario *layer freezing* pada EfficientNet-B3 (*Feature Extraction Only* 100%, *Shallow Fine-Tuning* 80%, dan *Moderate Fine-Tuning* 60%) lintas dua *dataset*, yaitu ODIR-5K dan Retinal Fundus 50K.

## Modul Utama
1. **Pengujian Citra Fundus**: Unggah citra retina (`.jpg`, `.png`, `.bmp`, `.webp`), pilih *dataset* referensi dan skenario *layer freezing*, lalu jalankan prapemrosesan identik pelatihan (koreksi EXIF, *black-margin cropping*, *letterbox padding* $300 \times 300$, normalisasi ImageNet) serta inferensi ONNX Runtime pada CPU.
2. **Dasbor Evaluasi**: Menyajikan tabel perbandingan metrik uji dengan interval kepercayaan 95% (*cluster bootstrap*), kurva pelatihan per *epoch*, *confusion matrix* $7 \times 7$, metrik per kelas, serta tolok ukur latensi ONNX.

## Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Gratis ke Streamlit Community Cloud + Hugging Face Model Hub

### 1. Unggah 6 Model `.onnx` ke Hugging Face Model Repo (Gratis)
1. Buat Model Repository publik baru di https://huggingface.co/new (misalnya `<username-hf>/efficientnet-b3-fundus-onnx`).
2. Buat token bertipe **Write** di https://huggingface.co/settings/tokens lalu login dari PowerShell:
```powershell
hf auth login
cd "D:\@KULIAH\1. Skripsi\Skripsiku\Website"
hf upload <username-hf>/efficientnet-b3-fundus-onnx odir5k_only_v4_aa52edfb9c31_onnx odir5k_only_v4_aa52edfb9c31_onnx --repo-type model
hf upload <username-hf>/efficientnet-b3-fundus-onnx fundus50k_leakage_fixed_v4_771d032fdf34_onnx fundus50k_leakage_fixed_v4_771d032fdf34_onnx --repo-type model
```

### 2. Push Kode Ringan ke GitHub (Gratis, < 1 MB)
Berkas `.onnx` dan `paket_data_uji/` sudah otomatis dikecualikan di `.gitignore`. Buat repositori publik baru di GitHub (misalnya `fundus-efficientnet-b3-web`), lalu jalankan dari folder `Website/`:
```powershell
cd "D:\@KULIAH\1. Skripsi\Skripsiku\Website"
git init
git add .
git commit -m "Initial commit: Streamlit fundus classifier and evaluation dashboard"
git branch -M main
git remote add origin https://github.com/<username-github>/fundus-efficientnet-b3-web.git
git push -u origin main
```

### 3. Deploy di Streamlit Community Cloud (Gratis)
1. Buka https://share.streamlit.io dan masuk memakai akun GitHub Anda.
2. Klik **Create app** -> pilih repo GitHub Anda, branch `main`, dan Main file path `app.py`.
3. Sebelum klik **Deploy**, klik **Advanced settings...** -> pada kolom **Secrets**, masukkan satu baris:
```toml
HF_MODEL_REPO = "<username-hf>/efficientnet-b3-fundus-onnx"
```
4. Klik **Save** lalu **Deploy**. Aplikasi akan terbit di `https://<nama-app>.streamlit.app`.

## Batas Penggunaan Klinis
Seluruh hasil prediksi bersifat alat bantu penelitian akademik dan tidak dapat menggantikan pemeriksaan maupun diagnosis dokter spesialis mata.
