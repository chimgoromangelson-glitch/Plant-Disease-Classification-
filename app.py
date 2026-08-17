from pathlib import Path
import json
import pandas as pd

import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriVision AI | Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# ORIGINAL TRAINED CNN
# ------------------------------------------------------------

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plant_disease_cnn.pth"
)


# ------------------------------------------------------------
# EFFICIENTNET-B0
# ------------------------------------------------------------

EFFICIENTNET_PATH = (
    BASE_DIR
    / "models"
    / "efficientnet_b0.pth"
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

RESULTS_PATH = (
    BASE_DIR
    / "results"
)

REPORT_PATH = (
    RESULTS_PATH
    / "classification_report.txt"
)

HISTORY_PATH = (
    RESULTS_PATH
    / "training_history.json"
)

TEST_METRICS_PATH = (
    RESULTS_PATH
    / "test_metrics.json"
)

ACCURACY_CURVE_PATH = (
    RESULTS_PATH
    / "accuracy_curve.png"
)

LOSS_CURVE_PATH = (
    RESULTS_PATH
    / "loss_curve.png"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_PATH
    / "confusion_matrix.png"
)


# ============================================================
# COMPARISON RESULTS
# ============================================================

COMPARISON_CSV_PATH = (
    RESULTS_PATH
    / "model_comparison.csv"
)

COMPARISON_CHART_PATH = (
    RESULTS_PATH
    / "model_comparison_chart.png"
)

ORIGINAL_CM_PATH = (
    RESULTS_PATH
    / "original_cnn_confusion_matrix.png"
)

EFFICIENTNET_CM_PATH = (
    RESULTS_PATH
    / "efficientnet_b0_confusion_matrix.png"
)

EFFICIENTNET_ACC_CURVE_PATH = (
    RESULTS_PATH
    / "efficientnet_accuracy_curve.png"
)

EFFICIENTNET_LOSS_CURVE_PATH = (
    RESULTS_PATH
    / "efficientnet_loss_curve.png"
)


# ============================================================
# ORIGINAL TEST DATASET
# ============================================================

TEST_DIR = (
    BASE_DIR
    / "dataset"
    / "split_data"
    / "test"
)


# ============================================================
# ORIGINAL PREPROCESSING
# ============================================================

IMAGE_SIZE = 128

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# ORIGINAL CLASSES
# ============================================================

DEFAULT_CLASSES = [
    "Pepper_Bacterial_Spot",
    "Potato_Early_Blight",
    "Tomato_Late_Blight"
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Pepper_Bacterial_Spot": {

        "plant": "Pepper",

        "icon": "🌶️",

        "name": "Pepper Bacterial Spot",

        "description":
            "A bacterial disease that produces dark "
            "or water-soaked spots and lesions on "
            "pepper leaves.",

        "symptoms":
            "Small dark spots, leaf lesions and "
            "progressive damage to affected leaves.",

        "advice":
            "Remove severely affected plant material, "
            "maintain field hygiene and avoid prolonged "
            "leaf wetness."
    },


    "Potato_Early_Blight": {

        "plant": "Potato",

        "icon": "🥔",

        "name": "Potato Early Blight",

        "description":
            "A fungal disease commonly associated "
            "with dark lesions on potato leaves.",

        "symptoms":
            "Dark brown lesions, sometimes with "
            "concentric ring patterns, followed by "
            "yellowing or leaf death.",

        "advice":
            "Remove badly affected leaves, maintain "
            "crop hygiene and reduce prolonged "
            "leaf moisture."
    },


    "Tomato_Late_Blight": {

        "plant": "Tomato",

        "icon": "🍅",

        "name": "Tomato Late Blight",

        "description":
            "A serious tomato disease that can cause "
            "rapidly expanding dark lesions on leaves.",

        "symptoms":
            "Dark irregular lesions, rapid leaf damage "
            "and browning of affected plant tissue.",

        "advice":
            "Remove severely affected material, improve "
            "air circulation and monitor plants regularly."
    }
}


# ============================================================
# UI STYLE — PREMIUM AGRIVISION DESIGN SYSTEM
# ============================================================
#
# Palette:
#   Primary   #1B4332  (deep forest green)
#   Secondary #2D6A4F  (leaf green)
#   Accent    #74C69D  (sage / mint)
#   Highlight #FFB703  (amber — warnings / CTA)
#   Danger    #E63946  (soft red)
#   Surface   #FFFFFF  (cards)
#   Backdrop  #F4F9F6  (app background)
#   Ink       #1B1B1B  (body text)
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #1B4332 !important;
        font-weight: 700 !important;
    }

    /* ---------- App background ---------- */
    .stApp {
        background: linear-gradient(180deg, #F4F9F6 0%, #EAF4EE 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12372A 0%, #1B4332 60%, #14532D 100%);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #F1FAEE !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
        transition: background 0.2s ease;
        border: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(116, 198, 157, 0.18);
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* ---------- Hero banner ---------- */
    .av-hero {
        background: linear-gradient(120deg, #1B4332 0%, #2D6A4F 55%, #40916C 100%);
        border-radius: 20px;
        padding: 2.4rem 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(27, 67, 50, 0.25);
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }

    .av-hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, transparent 70%);
        border-radius: 50%;
    }

    .av-hero h1 {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        margin-bottom: 0.3rem !important;
    }

    .av-hero p {
        color: #E9F5EE;
        font-size: 1.05rem;
        margin: 0;
        max-width: 640px;
    }

    .av-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.35);
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.9rem;
    }

    /* ---------- Section header ---------- */
    .av-section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #1B4332;
        font-size: 1.4rem;
        margin: 1.6rem 0 0.6rem 0;
        border-left: 5px solid #74C69D;
        padding-left: 12px;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #DCEEE3;
        border-radius: 16px;
        padding: 16px 14px;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(27, 67, 50, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #52796F !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1B4332 !important;
        font-weight: 800 !important;
    }

    /* ---------- Containers / bordered cards ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(27, 67, 50, 0.07);
        background: #ffffff;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(120deg, #2D6A4F, #40916C);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.25);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(45, 106, 79, 0.35);
        filter: brightness(1.05);
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #95D5B2;
        border-radius: 16px;
        padding: 0.8rem;
        background: #FBFFFD;
    }

    /* ---------- Progress bar ---------- */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2D6A4F, #74C69D) !important;
    }

    /* ---------- Alerts (info/success/warning/error) ---------- */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        font-weight: 500;
    }

    /* ---------- Dataframe / table ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #DCEEE3;
    }

    /* ---------- Tabs / radio pills for navigation feel ---------- */
    .stRadio > div {
        gap: 4px;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: #DCEEE3 !important;
    }

    /* ---------- Footer ---------- */
    .av-footer {
        text-align: center;
        color: #52796F;
        font-size: 0.85rem;
        padding-top: 0.5rem;
    }

    /* ---------- Code / classification report ---------- */
    .stCodeBlock, pre {
        border-radius: 12px !important;
    }

    /* ---------- Images ---------- */
    [data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)


def render_hero(badge_text, title, subtitle):
    """Reusable styled hero banner for each page."""

    st.markdown(
        f"""
        <div class="av-hero">
            <span class="av-badge">{badge_text}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_title(text):
    """Reusable styled section subtitle."""

    st.markdown(
        f'<div class="av-section-title">{text}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ORIGINAL CNN ARCHITECTURE
# ============================================================

class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64,
            128,
            3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.adaptive_pool = (
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x):

        x = self.pool(
            torch.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool(
            torch.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool(
            torch.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = self.adaptive_pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        return self.fc2(x)


# ============================================================
# LOAD ORIGINAL CNN
# ============================================================

@st.cache_resource
def load_original_cnn():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Original CNN model not found:\n"
            f"{MODEL_PATH}"
        )


    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Invalid original CNN checkpoint."
        )


    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Original CNN checkpoint does not "
            "contain model_state_dict."
        )


    class_names = list(
        checkpoint.get(
            "class_names",
            DEFAULT_CLASSES
        )
    )


    image_size = int(
        checkpoint.get(
            "image_size",
            IMAGE_SIZE
        )
    )


    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )


    # Remove DataParallel prefix if present

    if any(
        key.startswith("module.")
        for key in state_dict
    ):

        state_dict = {

            key.replace(
                "module.",
                "",
                1
            ): value

            for key, value
            in state_dict.items()
        }


    model = PlantDiseaseCNN(
        len(class_names)
    )


    model.load_state_dict(
        state_dict,
        strict=True
    )


    model.eval()


    return (
        model,
        class_names,
        image_size
    )


# ============================================================
# LOAD ORIGINAL CNN
# ============================================================

original_model = None
original_model_error = None


try:

    (
        original_model,
        class_names,
        image_size
    ) = load_original_cnn()

except Exception as exc:

    class_names = (
        DEFAULT_CLASSES.copy()
    )

    image_size = IMAGE_SIZE

    original_model_error = exc


# ============================================================
# IMAGE PREPROCESSING
# SAME AS ORIGINAL CODE
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (
            image_size,
            image_size
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )
])


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

@st.cache_resource
def load_efficientnet():

    if not EFFICIENTNET_PATH.exists():

        return (
            None,
            f"EfficientNet-B0 model not found:\n"
            f"{EFFICIENTNET_PATH}"
        )


    try:

        checkpoint = torch.load(
            EFFICIENTNET_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            EFFICIENTNET_PATH,
            map_location="cpu"
        )


    # Create EfficientNet-B0 architecture

    model = models.efficientnet_b0(
        weights=None
    )


    input_features = (
        model.classifier[1].in_features
    )


    # Three disease classes

    model.classifier[1] = nn.Linear(
        input_features,
        len(class_names)
    )


    # Get saved weights

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif isinstance(
        checkpoint,
        dict
    ):

        state_dict = checkpoint

    else:

        return (
            None,
            "Invalid EfficientNet-B0 checkpoint."
        )


    # Remove DataParallel prefix

    if any(
        key.startswith("module.")
        for key in state_dict
    ):

        state_dict = {

            key.replace(
                "module.",
                "",
                1
            ): value

            for key, value
            in state_dict.items()
        }


    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except Exception as exc:

        return (
            None,
            "EfficientNet-B0 checkpoint "
            "does not match the expected "
            "three-class architecture.\n\n"
            + str(exc)
        )


    model.eval()


    return (
        model,
        None
    )


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

efficientnet_model = None
efficientnet_error = None


try:

    (
        efficientnet_model,
        efficientnet_error
    ) = load_efficientnet()

except Exception as exc:

    efficientnet_error = exc


# ============================================================
# ORIGINAL CNN TEST ACCURACY
# ============================================================

@st.cache_data
def load_original_test_metrics():

    # First try saved test metrics

    if TEST_METRICS_PATH.exists():

        try:

            return json.loads(
                TEST_METRICS_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            pass


    # Otherwise calculate from test dataset

    if (
        original_model is None
        or
        not TEST_DIR.exists()
    ):

        return None


    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=image_transform
    )


    if (
        test_dataset.classes
        != class_names
    ):

        return None


    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in loader:

            outputs = original_model(
                images
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    accuracy = (
        correct / total
        if total > 0
        else 0
    )


    return {

        "test_accuracy":
            accuracy,

        "correct_predictions":
            correct,

        "test_images":
            total
    }


original_test_metrics = (
    load_original_test_metrics()
)


# ============================================================
# LOAD MODEL COMPARISON RESULTS
# ============================================================

@st.cache_data
def load_comparison_results():

    if not COMPARISON_CSV_PATH.exists():

        return None


    try:

        return pd.read_csv(
            COMPARISON_CSV_PATH
        )

    except Exception:

        return None


comparison_df = (
    load_comparison_results()
)


# ============================================================
# ORIGINAL CNN PREDICTION
# ============================================================

def predict_original(image):

    tensor = (
        image_transform(image)
        .unsqueeze(0)
    )


    with torch.no_grad():

        outputs = original_model(
            tensor
        )

        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )[0]
        )

        pred_idx = int(
            torch.argmax(
                probabilities
            ).item()
        )


    return (

        class_names[pred_idx],

        float(
            probabilities[
                pred_idx
            ].item()
        ),

        probabilities
    )


# ============================================================
# EFFICIENTNET PREDICTION
# ============================================================

def predict_efficientnet(image):

    tensor = (
        image_transform(image)
        .unsqueeze(0)
    )


    with torch.no_grad():

        outputs = (
            efficientnet_model(
                tensor
            )
        )

        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )[0]
        )

        pred_idx = int(
            torch.argmax(
                probabilities
            ).item()
        )


    return (

        class_names[pred_idx],

        float(
            probabilities[
                pred_idx
            ].item()
        ),

        probabilities
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding: 0.4rem 0 0.8rem 0;">
            <div style="font-size: 2.4rem; line-height: 1;">🌿</div>
            <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.3rem; margin-top:0.3rem;">
                AgriVision AI
            </div>
            <div style="opacity:0.75; font-size:0.85rem;">Intelligent Leaf Classification</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Disease Detection",
            "📊 Model Performance",
            "⚖️ Model Comparison",
            "📚 Disease Guide",
            "ℹ️ About"
        ]
    )


    st.divider()


    st.subheader(
        "Supported Classes"
    )


    for name in class_names:

        info = DISEASE_INFO.get(
            name,
            {}
        )

        st.write(
            f"{info.get('icon', '🌿')} "
            f"{info.get('name', name)}"
        )


    st.divider()


    if original_model is not None:

        st.success(
            "Original CNN loaded"
        )

    else:

        st.error(
            "Original CNN not loaded"
        )


    if efficientnet_model is not None:

        st.success(
            "EfficientNet-B0 loaded"
        )

    else:

        st.warning(
            "EfficientNet-B0 not loaded"
        )


    st.caption(
        "PyTorch • CNN • EfficientNet • Streamlit"
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    render_hero(
        "AI DECISION SUPPORT",
        "🌿 AgriVision AI",
        "Deep learning-powered classification of Pepper, Potato and "
        "Tomato leaf diseases — built to help growers spot problems early."
    )


    st.info(
        "This application retains the original "
        "custom CNN and adds EfficientNet-B0 "
        "for comparative analysis."
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Original CNN",
            "Ready"
            if original_model is not None
            else "Not Loaded"
        )


    with m2:

        st.metric(
            "EfficientNet-B0",
            "Ready"
            if efficientnet_model is not None
            else "Not Loaded"
        )


    with m3:

        st.metric(
            "Classes",
            len(class_names)
        )


    with m4:

        if original_test_metrics:

            st.metric(
                "CNN Test Accuracy",
                f"{original_test_metrics['test_accuracy'] * 100:.2f}%"
            )

        else:

            st.metric(
                "CNN Test Accuracy",
                "Not available"
            )


    render_section_title(
        "Supported Plants"
    )


    cols = st.columns(3)


    for col, name in zip(
        cols,
        class_names
    ):

        info = DISEASE_INFO.get(
            name,
            {}
        )


        with col:

            with st.container(
                border=True
            ):

                st.subheader(
                    f"{info.get('icon', '🌿')} "
                    f"{info.get('plant', name)}"
                )

                st.write(
                    info.get(
                        "name",
                        name.replace(
                            "_",
                            " "
                        )
                    )
                )


# ============================================================
# DISEASE DETECTION
# ORIGINAL 3-PHOTO FEATURE RETAINED
# ============================================================

elif page == "🔍 Disease Detection":

    render_hero(
        "INSTANT ANALYSIS",
        "🔍 Disease Detection",
        "Upload up to 3 leaf photos at once. Each image is analysed "
        "independently by your original trained CNN."
    )


    if original_model is None:

        st.error(
            "The trained CNN could not be loaded."
        )


        with st.expander(
            "Show technical error"
        ):

            st.code(
                str(
                    original_model_error
                )
            )


    else:

        uploaded_files = st.file_uploader(
            "Choose up to 3 leaf images",

            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],

            accept_multiple_files=True,

            help=(
                "Select one, two or three images. "
                "Maximum: 3 images per analysis."
            )
        )


        if len(uploaded_files) > 3:

            st.error(
                "Please select a maximum of 3 images."
            )

            uploaded_files = (
                uploaded_files[:3]
            )


        if uploaded_files:

            st.info(
                f"{len(uploaded_files)} "
                f"image(s) selected."
            )


            result_cols = st.columns(
                len(uploaded_files)
            )


            for col, uploaded_file in zip(
                result_cols,
                uploaded_files
            ):

                with col:

                    image = Image.open(
                        uploaded_file
                    ).convert("RGB")


                    st.image(
                        image,
                        caption=uploaded_file.name,
                        use_container_width=True
                    )


                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_original(
                        image
                    )


                    info = DISEASE_INFO.get(
                        predicted_class,
                        {}
                    )


                    st.success(
                        f"{info.get('icon', '🌿')} "
                        f"{info.get('name', predicted_class)}"
                    )


                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )


                    st.progress(
                        confidence
                    )


                    st.caption(
                        "Class probabilities"
                    )


                    for i, name in enumerate(
                        class_names
                    ):

                        label = (
                            DISEASE_INFO
                            .get(name, {})
                            .get(
                                "name",
                                name
                            )
                        )


                        st.write(
                            f"{label}: "
                            f"{probabilities[i].item() * 100:.2f}%"
                        )


                    with st.expander(
                        "Disease information"
                    ):

                        st.write(
                            f"**Symptoms:** "
                            f"{info.get('symptoms', 'Not available')}"
                        )


                        st.write(
                            f"**Suggested action:** "
                            f"{info.get('advice', 'Not available')}"
                        )


        else:

            st.info(
                "Upload up to three images "
                "to begin prediction."
            )


# ============================================================
# ORIGINAL MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    render_hero(
        "TRAINING INSIGHTS",
        "📊 Model Performance",
        "Evaluation information from your original trained CNN, "
        "including accuracy, loss curves and the confusion matrix."
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Input Size",
        f"{image_size} × {image_size}"
    )


    c2.metric(
        "Classes",
        len(class_names)
    )


    c3.metric(
        "Model",
        "Original CNN"
    )


    if original_test_metrics:

        c4.metric(
            "Test Accuracy",
            f"{original_test_metrics['test_accuracy'] * 100:.2f}%"
        )

    else:

        c4.metric(
            "Test Accuracy",
            "Not available"
        )


    if original_test_metrics:

        st.success(
            f"Test Accuracy: "
            f"{original_test_metrics['test_accuracy'] * 100:.2f}% "
            f"("
            f"{original_test_metrics.get('correct_predictions', '?')} "
            f"correct out of "
            f"{original_test_metrics.get('test_images', '?')} "
            f"test images)."
        )


    else:

        st.warning(
            "Test accuracy is not available."
        )


    if HISTORY_PATH.exists():

        try:

            history = json.loads(
                HISTORY_PATH.read_text(
                    encoding="utf-8"
                )
            )


            train_acc = history.get(
                "train_accuracy",
                []
            )


            val_acc = history.get(
                "val_accuracy",
                []
            )


            if train_acc and val_acc:

                a, b = st.columns(2)


                a.metric(
                    "Best Training Accuracy",
                    f"{max(train_acc) * 100:.2f}%"
                )


                b.metric(
                    "Best Validation Accuracy",
                    f"{max(val_acc) * 100:.2f}%"
                )


        except Exception:

            pass


    render_section_title(
        "Classification Report"
    )


    if REPORT_PATH.exists():

        st.code(
            REPORT_PATH.read_text(
                encoding="utf-8"
            ),
            language="text"
        )

    else:

        st.info(
            "classification_report.txt "
            "was not found in results."
        )


    p1, p2 = st.columns(2)


    with p1:

        if ACCURACY_CURVE_PATH.exists():

            st.image(
                str(
                    ACCURACY_CURVE_PATH
                ),
                caption=(
                    "Training and Validation Accuracy"
                ),
                use_container_width=True
            )


    with p2:

        if LOSS_CURVE_PATH.exists():

            st.image(
                str(
                    LOSS_CURVE_PATH
                ),
                caption=(
                    "Training and Validation Loss"
                ),
                use_container_width=True
            )


    if CONFUSION_MATRIX_PATH.exists():

        st.image(
            str(
                CONFUSION_MATRIX_PATH
            ),
            caption=(
                "Original CNN Confusion Matrix"
            ),
            use_container_width=True
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "⚖️ Model Comparison":

    render_hero(
        "HEAD-TO-HEAD",
        "⚖️ Model Comparison",
        "Comparative analysis of the original custom CNN and "
        "EfficientNet-B0 for plant disease classification."
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    render_section_title(
        "🤖 Model Status"
    )


    s1, s2 = st.columns(2)


    with s1:

        if original_model is not None:

            st.success(
                "Original CNN: Loaded"
            )

        else:

            st.error(
                "Original CNN: Not Loaded"
            )


    with s2:

        if efficientnet_model is not None:

            st.success(
                "EfficientNet-B0: Loaded"
            )

        else:

            st.warning(
                "EfficientNet-B0: Not Loaded"
            )

            if efficientnet_error:

                with st.expander(
                    "Show EfficientNet error"
                ):

                    st.code(
                        str(
                            efficientnet_error
                        )
                    )


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    render_section_title(
        "📊 Test Performance Comparison"
    )


    if comparison_df is not None:

        display_df = (
            comparison_df.copy()
        )


        percentage_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]


        for column in percentage_columns:

            if column in display_df.columns:

                display_df[column] = (

                    pd.to_numeric(
                        display_df[column],
                        errors="coerce"
                    )

                    * 100

                ).round(2).map(

                    lambda value:
                    f"{value:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "model_comparison.csv was not found."
        )


        st.info(
            "Run this first:\n\n"
            "python compare_models.py"
        )


    # --------------------------------------------------------
    # ACCURACY METRICS
    # --------------------------------------------------------

    if comparison_df is not None:

        accuracy_column = None


        for candidate in [
            "Accuracy",
            "Test Accuracy"
        ]:

            if candidate in comparison_df.columns:

                accuracy_column = candidate

                break


        if accuracy_column:

            render_section_title(
                "🎯 Test Accuracy"
            )


            metric_columns = st.columns(
                len(comparison_df)
            )


            for col, (_, row) in zip(
                metric_columns,
                comparison_df.iterrows()
            ):

                model_name = str(
                    row.get(
                        "Model",
                        "Model"
                    )
                )


                try:

                    accuracy = float(
                        row[
                            accuracy_column
                        ]
                    )


                    # If stored as decimal

                    if accuracy <= 1:

                        accuracy *= 100


                    col.metric(
                        model_name,
                        f"{accuracy:.2f}%"
                    )


                except Exception:

                    col.metric(
                        model_name,
                        "N/A"
                    )


    # --------------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------------

    render_section_title(
        "📈 Performance Comparison"
    )


    if COMPARISON_CHART_PATH.exists():

        st.image(
            str(
                COMPARISON_CHART_PATH
            ),
            caption=(
                "Original CNN vs EfficientNet-B0"
            ),
            use_container_width=True
        )


    else:

        st.info(
            "model_comparison_chart.png "
            "was not found."
        )


    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    render_section_title(
        "🔲 Confusion Matrix Comparison"
    )


    cm1, cm2 = st.columns(2)


    with cm1:

        st.markdown(
            "### 🧠 Original CNN"
        )


        if ORIGINAL_CM_PATH.exists():

            st.image(
                str(
                    ORIGINAL_CM_PATH
                ),
                use_container_width=True
            )


        elif CONFUSION_MATRIX_PATH.exists():

            st.image(
                str(
                    CONFUSION_MATRIX_PATH
                ),
                use_container_width=True
            )


        else:

            st.info(
                "Original CNN confusion matrix "
                "not found."
            )


    with cm2:

        st.markdown(
            "### ⚡ EfficientNet-B0"
        )


        if EFFICIENTNET_CM_PATH.exists():

            st.image(
                str(
                    EFFICIENTNET_CM_PATH
                ),
                use_container_width=True
            )


        else:

            st.info(
                "EfficientNet-B0 confusion matrix "
                "not found."
            )


    # --------------------------------------------------------
    # EFFICIENTNET TRAINING CURVES
    # --------------------------------------------------------

    render_section_title(
        "📚 EfficientNet-B0 Training History"
    )


    e1, e2 = st.columns(2)


    with e1:

        if EFFICIENTNET_ACC_CURVE_PATH.exists():

            st.image(
                str(
                    EFFICIENTNET_ACC_CURVE_PATH
                ),
                caption=(
                    "EfficientNet-B0 Accuracy"
                ),
                use_container_width=True
            )


    with e2:

        if EFFICIENTNET_LOSS_CURVE_PATH.exists():

            st.image(
                str(
                    EFFICIENTNET_LOSS_CURVE_PATH
                ),
                caption=(
                    "EfficientNet-B0 Loss"
                ),
                use_container_width=True
            )


    # --------------------------------------------------------
    # SAME IMAGE COMPARISON
    # --------------------------------------------------------

    st.divider()


    render_section_title(
        "🔬 Compare Both Models on the Same Images"
    )


    st.write(
        "Upload up to 3 leaf images. "
        "Both models will analyse the same images."
    )


    comparison_images = st.file_uploader(
        "Choose up to 3 leaf images",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],

        accept_multiple_files=True,

        key="comparison_images"
    )


    if len(comparison_images) > 3:

        st.error(
            "Please select a maximum of 3 images."
        )

        comparison_images = (
            comparison_images[:3]
        )


    if comparison_images:

        for uploaded_file in comparison_images:

            st.divider()


            image = Image.open(
                uploaded_file
            ).convert("RGB")


            st.image(
                image,
                caption=uploaded_file.name,
                width=400
            )


            c1, c2 = st.columns(2)


            # ------------------------------------------------
            # ORIGINAL CNN
            # ------------------------------------------------

            with c1:

                st.markdown(
                    "### 🧠 Original CNN"
                )


                if original_model is not None:

                    (
                        cnn_class,
                        cnn_confidence,
                        cnn_probabilities
                    ) = predict_original(
                        image
                    )


                    info = DISEASE_INFO.get(
                        cnn_class,
                        {}
                    )


                    st.success(
                        f"{info.get('icon', '🌿')} "
                        f"{info.get('name', cnn_class)}"
                    )


                    st.metric(
                        "Confidence",
                        f"{cnn_confidence * 100:.2f}%"
                    )


                    st.progress(
                        cnn_confidence
                    )


                else:

                    st.error(
                        "Original CNN is not loaded."
                    )


            # ------------------------------------------------
            # EFFICIENTNET
            # ------------------------------------------------

            with c2:

                st.markdown(
                    "### ⚡ EfficientNet-B0"
                )


                if efficientnet_model is not None:

                    (
                        eff_class,
                        eff_confidence,
                        eff_probabilities
                    ) = predict_efficientnet(
                        image
                    )


                    info = DISEASE_INFO.get(
                        eff_class,
                        {}
                    )


                    st.success(
                        f"{info.get('icon', '🌿')} "
                        f"{info.get('name', eff_class)}"
                    )


                    st.metric(
                        "Confidence",
                        f"{eff_confidence * 100:.2f}%"
                    )


                    st.progress(
                        eff_confidence
                    )


                else:

                    st.error(
                        "EfficientNet-B0 is not loaded."
                    )


            # ------------------------------------------------
            # AGREEMENT
            # ------------------------------------------------

            if (
                original_model is not None
                and
                efficientnet_model is not None
            ):

                st.subheader(
                    "🔎 Prediction Agreement"
                )


                if cnn_class == eff_class:

                    st.success(
                        "✅ Both models predicted "
                        "the same disease."
                    )


                else:

                    st.warning(
                        "⚠️ The two models predicted "
                        "different diseases."
                    )


                    a1, a2 = st.columns(2)


                    with a1:

                        st.write(
                            "Original CNN: "
                            f"**{DISEASE_INFO.get(cnn_class, {}).get('name', cnn_class)}**"
                        )


                    with a2:

                        st.write(
                            "EfficientNet-B0: "
                            f"**{DISEASE_INFO.get(eff_class, {}).get('name', eff_class)}**"
                        )


    else:

        st.info(
            "Upload one, two or three images "
            "to compare both models."
        )


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.divider()


    render_section_title(
        "🧠 Models Being Compared"
    )


    left, right = st.columns(2)


    with left:

        with st.container(border=True):

            st.markdown(
                """
### 🧠 Original Custom CNN

- Your original trained CNN
- 3 convolutional layers
- Batch Normalization
- Max Pooling
- Adaptive Average Pooling
- Fully connected layer
- Dropout
- Trained for the 3 plant disease classes
"""
            )


    with right:

        with st.container(border=True):

            st.markdown(
                """
### ⚡ EfficientNet-B0

- Modern CNN architecture
- Used as the comparison model
- Trained/fine-tuned for the same 3 classes
- Evaluated on the same test dataset
- Compared using Accuracy, Precision, Recall and F1-score
"""
            )


# ============================================================
# DISEASE GUIDE
# ============================================================

elif page == "📚 Disease Guide":

    render_hero(
        "KNOWLEDGE BASE",
        "📚 Disease Guide",
        "Symptoms and recommended actions for each supported "
        "disease class."
    )


    for name in class_names:

        info = DISEASE_INFO.get(
            name,
            {}
        )


        with st.container(
            border=True
        ):

            st.subheader(
                f"{info.get('icon', '🌿')} "
                f"{info.get('name', name)}"
            )


            st.write(
                f"**Plant:** "
                f"{info.get('plant', 'Unknown')}"
            )


            st.write(
                f"**Description:** "
                f"{info.get('description', 'Not available')}"
            )


            st.write(
                f"**Symptoms:** "
                f"{info.get('symptoms', 'Not available')}"
            )


            st.write(
                f"**Suggested Action:** "
                f"{info.get('advice', 'Not available')}"
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_hero(
        "PROJECT INFO",
        "ℹ️ About AgriVision AI",
        "A deep-learning computer-vision application for multiclass "
        "plant disease image classification."
    )


    st.info(
        "The application retains the original custom CNN "
        "and adds EfficientNet-B0 for comparative analysis."
    )


    a, b = st.columns(2)


    with a:

        with st.container(border=True):

            st.subheader(
                "Technology"
            )


            st.write(
                "**Original Model:** "
                "Custom Convolutional Neural Network"
            )


            st.write(
                "**Comparison Model:** "
                "EfficientNet-B0"
            )


            st.write(
                "**Framework:** PyTorch"
            )


            st.write(
                "**Interface:** Streamlit"
            )


            st.write(
                f"**Input:** "
                f"{image_size} × {image_size} pixels"
            )


    with b:

        with st.container(border=True):

            st.subheader(
                "Supported Classes"
            )


            for name in class_names:

                info = DISEASE_INFO.get(
                    name,
                    {}
                )


                st.write(
                    f"{info.get('icon', '🌿')} "
                    f"{info.get('name', name)}"
                )


    st.warning(
        "This application is an AI decision-support "
        "tool and should not replace professional "
        "agricultural diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="av-footer">🌿 AgriVision AI • Deep Learning • '
    'Computer Vision • PyTorch • Streamlit</div>',
    unsafe_allow_html=True
)
