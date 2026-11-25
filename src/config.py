from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

# Set via Streamlit Secrets (HF_MODEL_REPO = "username/repo-name") when deploying to Streamlit Cloud.
DEFAULT_HF_MODEL_REPO = ""

LABELS = ["N", "D", "G", "C", "A", "H", "M"]
IMAGE_SIZE = 300
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class ClassInfo:
    code: str
    name_id: str
    name_en: str
    visual_cue: str


CLASS_INFO: dict[str, ClassInfo] = {
    "N": ClassInfo(
        code="N",
        name_id="Normal",
        name_en="Normal",
        visual_cue="Pembuluh darah tajam, batas diskus optikus tegas, makula bersih tanpa lesi.",
    ),
    "D": ClassInfo(
        code="D",
        name_id="Retinopati Diabetik",
        name_en="Diabetic Retinopathy",
        visual_cue="Mikroaneurisma, perdarahan retina titik atau bercak, serta eksudat keras/lunak.",
    ),
    "G": ClassInfo(
        code="G",
        name_id="Glaukoma",
        name_en="Glaucoma",
        visual_cue="Pembesaran cup-to-disc ratio dan penipisan jaringan tepi neuroretina.",
    ),
    "C": ClassInfo(
        code="C",
        name_id="Katarak",
        name_en="Cataract",
        visual_cue="Tampilan citra berkabut atau buram menyeluruh akibat kekeruhan lensa mata.",
    ),
    "A": ClassInfo(
        code="A",
        name_id="Degenerasi Makula (AMD)",
        name_en="Age-Related Macular Degeneration",
        visual_cue="Endapan ekstraseluler kekuningan (drusen) dan perubahan epitel pigmen di makula.",
    ),
    "H": ClassInfo(
        code="H",
        name_id="Retinopati Hipertensi",
        name_en="Hypertensive Retinopathy",
        visual_cue="Penyempitan arteriol retina, perubahan persilangan arteriovenosa, perdarahan, dan eksudat.",
    ),
    "M": ClassInfo(
        code="M",
        name_id="Miopia Patologis",
        name_en="Pathological Myopia",
        visual_cue="Pola tigroid fundus, tilted disc, bercak Fuchs, dan lacquer cracks pada membran Bruch.",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    short_label: str
    onnx_dir: Path
    data_dir: Path
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "odir5k": DatasetSpec(
        key="odir5k",
        label="ODIR-5K (4.550 citra bersih)",
        short_label="ODIR-5K",
        onnx_dir=BASE_DIR / "odir5k_only_v4_aa52edfb9c31_onnx",
        data_dir=DATA_DIR / "odir5k",
        description="Data klinis multipusat dari 2.358 pasien unik dengan ketimpangan kelas tinggi.",
    ),
    "fundus50k": DatasetSpec(
        key="fundus50k",
        label="Retinal Fundus 50K (57.066 citra bersih)",
        short_label="Retinal Fundus 50K",
        onnx_dir=BASE_DIR / "fundus50k_leakage_fixed_v4_771d032fdf34_onnx",
        data_dir=DATA_DIR / "fundus50k",
        description="Dataset skala besar dari 6.573 grup pasien unik setelah pemisahan ulang bebas kebocoran data.",
    ),
}


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    label: str
    short_label: str
    freeze_target: str
    onnx_filename: str
    block_note: str


SCENARIOS: dict[str, ScenarioSpec] = {
    "feature_extraction": ScenarioSpec(
        key="feature_extraction",
        label="Feature Extraction Only (100% Beku)",
        short_label="Feature Extraction (100%)",
        freeze_target="100%",
        onnx_filename="model_feature_extraction.onnx",
        block_note="28 dari 28 blok backbone dibekukan; hanya classifier head yang dilatih.",
    ),
    "shallow": ScenarioSpec(
        key="shallow",
        label="Shallow Fine-Tuning (80% Beku)",
        short_label="Shallow Fine-Tuning (80%)",
        freeze_target="80%",
        onnx_filename="model_shallow.onnx",
        block_note="22 dari 28 blok awal dibekukan; 6 blok teratas dan classifier head aktif.",
    ),
    "moderate": ScenarioSpec(
        key="moderate",
        label="Moderate Fine-Tuning (60% Beku)",
        short_label="Moderate Fine-Tuning (60%)",
        freeze_target="60%",
        onnx_filename="model_moderate.onnx",
        block_note="17 dari 28 blok awal dibekukan; 11 blok teratas dan classifier head aktif.",
    ),
}

SCENARIO_ORDER = ["feature_extraction", "shallow", "moderate"]
