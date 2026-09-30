import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gdown
import os
import base64
from io import BytesIO
from openai import OpenAI
from typing import Any, cast
import httpx
import matplotlib.cm as cm


# ==================================================
# STREAMLIT SETTINGS
# ==================================================

st.set_page_config(
    page_title="AI E-Commerce Product Generator",
    page_icon="🛍️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 10px;
    }

    .result-label {
        font-size: 14px;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 26px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .description-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        font-size: 17px;
        line-height: 1.6;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🛍️ AI E-Commerce Product Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ürün görsellerini derin öğrenme ve LLM teknolojileriyle analiz edin.
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CATEGORY MAPPING
# ==================================================

CATEGORY_TR = {
    "Tshirts": "tişört",
    "Shirts": "gömlek",
    "Casual Shoes": "günlük ayakkabı",
    "Sports Shoes": "spor ayakkabı",
    "Formal Shoes": "klasik ayakkabı",
    "Watches": "saat",
    "Handbags": "el çantası",
    "Heels": "topuklu ayakkabı",
    "Jeans": "kot pantolon",
    "Kurtas": "kurta",
    "Perfume and Body Mist": "parfüm",
    "Sandals": "sandalet",
    "Socks": "çorap",
    "Sunglasses": "güneş gözlüğü",
    "Tops": "üst giyim",
    "Wallets": "cüzdan",
    "Backpacks": "sırt çantası",
    "Belts": "kemer",
    "Briefs": "boxer",
    "Flip Flops": "terlik",
}


# ==================================================
# MODEL SETTINGS
# ==================================================

MODEL_DRIVE_ID = "14oOXk7DLQ6VFDMa9-Hg4ix6neP8IW6wJ"
MODEL_PATH = "baseline_resnet50.pth"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner(
            "Model indiriliyor, bu ilk açılışta biraz sürebilir..."
        ):

            gdown.download(
                id=MODEL_DRIVE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    num_classes = len(
        checkpoint["class_to_idx"]
    )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    idx_to_class = checkpoint["idx_to_class"]

    return model, idx_to_class, device


model, idx_to_class, device = load_model()


# ==================================================
# GRAD-CAM
# ==================================================

activations = {}
gradients = {}


def forward_hook(module, input, output):
    activations["value"] = output.detach()


def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()


target_layer = model.layer4

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


def generate_gradcam(
    input_tensor,
    target_class
):

    model.zero_grad()

    output = model(
        input_tensor
    )

    score = output[
        0,
        target_class
    ]

    score.backward()

    act = activations["value"][0]

    grad = gradients["value"][0]

    weights = grad.mean(
        dim=(1, 2)
    )

    cam = torch.zeros(
        act.shape[1:],
        device=device
    )

    for i, w in enumerate(weights):

        cam += w * act[i]

    cam = torch.relu(
        cam
    )

    cam = cam.cpu().numpy()

    cam = cv2.resize(
        cam,
        (IMG_SIZE, IMG_SIZE)
    )

    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min()
        + 1e-8
    )

    return cam


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

IMG_SIZE = 224

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


# ==================================================
# IMAGE → BASE64
# ==================================================

def image_to_base64(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📤 Ürün görseli yükleyin",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ==================================================
    # RESNET PREDICTION
    # ==================================================

    input_tensor = eval_transform(
        image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred_idx = torch.max(
            probs,
            1
        )


    predicted_category_en = idx_to_class[
        pred_idx.item()
    ]

    predicted_category_tr = CATEGORY_TR.get(
        predicted_category_en,
        predicted_category_en
    )

    confidence_value = confidence.item()


    # ==================================================
    # IMAGE + PREDICTION
    # ==================================================

    st.markdown(
        '<div class="section-title">🎯 Model Tahmini</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.image(
            image,
            caption="Yüklenen ürün görseli"
        )


    with col2:
        st.write("**Tahmin edilen kategori**")
        st.write(predicted_category_tr)

        st.write("**Model güven skoru**")
        st.write(f"{confidence_value:.0%}")
        
    # ==================================================
    # GRAD-CAM
    # ==================================================

    st.markdown(
        '<div class="section-title">🔍 Model Neye Bakıyor?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Grad-CAM, modelin tahmin sırasında görselin hangi "
        "bölgelerine daha fazla odaklandığını gösterir."
    )


    cam = generate_gradcam(
        input_tensor,
        pred_idx.item()
    )


    original_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    original_np = (
        np.array(
            original_resized
        )
        / 255.0
    )


    heatmap = cm.jet(
        cam
    )[:, :, :3]


    overlay = (
        0.5 * original_np
        + 0.5 * heatmap
    )


    col3, col4 = st.columns(2)


    with col3:

        st.image(
            heatmap,
            caption="Grad-CAM Isı Haritası",
            clamp=True
        )


    with col4:

        st.image(
            overlay,
            caption="Orijinal Görsel + Grad-CAM",
            clamp=True
        )


    # ==================================================
    # LLM PRODUCT DESCRIPTION
    # ==================================================

    st.markdown(
        '<div class="section-title">✍️ Otomatik Ürün Açıklaması</div>',
        unsafe_allow_html=True
    )


    api_key = st.secrets.get(
        "OPENAI_API_KEY"
    )


    if api_key:

        client = OpenAI(
            api_key=api_key,
            http_client=cast(
                Any,
                httpx.Client()
            )
        )


        info = {
            "predicted_category":
                predicted_category_tr,

            "confidence":
                confidence_value
        }


        details = [
            f"Kategori: {info['predicted_category']}"
        ]


        confidence_note = ""


        if info["confidence"] < 0.6:

            confidence_note = """
- Kategori tahmini kesin değil.
- Bu nedenle kategori adını kesin bir iddia gibi sunma.
- Daha genel bir ifade kullan.
"""


        details_text = "\n".join(
            details
        )


        prompt = f"""
Bir e-ticaret sitesi için ürün görseline dayalı
çok kısa ve doğal bir ürün açıklaması yaz.

Model tarafından tahmin edilen kategori:
{details_text}

Görseli dikkatlice incele.

Kesin kurallar:

- Yalnızca görselde açıkça görülebilen özellikleri belirt.
- Renk, desen, baskı, şekil, kol tipi veya görünür tasarım
  detaylarını yalnızca gerçekten görselde varsa belirt.
- Görseldeki yazı, sembol veya logonun hangi markaya ait
  olduğunu kesin olarak belirleyemiyorsan marka adı verme.
- Bir logonun varlığından emin değilsen logo olduğunu söyleme.
- Görselde doğrulanamayan hiçbir özelliği uydurma.
- Marka, fiyat, kumaş, malzeme, beden, sezon veya teknik
  özelliklerden kesin olarak anlaşılamıyorsa bahsetme.
- "rahat", "şık", "zarif", "kaliteli", "modern",
  "ideal", "dayanıklı", "yumuşak" gibi öznel pazarlama
  ifadelerini kullanma.
- "günlük kullanım", "her mevsim", "farklı kombinler"
  gibi görselden çıkarılamayan ifadeler kullanma.
- Modelin confidence değerinden bahsetme.
- Yapay zeka veya model tahmininden bahsetme.
- En fazla 2 kısa cümle yaz.
- Sade ve doğal Türkçe kullan.

{confidence_note}
"""


        image_base64 = image_to_base64(
            image
        )


        with st.spinner(
            "Görsel analiz ediliyor ve ürün açıklaması oluşturuluyor..."
        ):

            response = client.chat.completions.create(

                model="gpt-4o-mini",

                messages=[

                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "text",
                                "text": prompt
                            },

                            {
                                "type": "image_url",

                                "image_url": {

                                    "url":
                                        "data:image/jpeg;base64,"
                                        + image_base64
                                }
                            }

                        ]
                    }

                ],

                temperature=0.2
            )


        description = (
            response
            .choices[0]
            .message
            .content
        )


        st.markdown(
            f"""
            <div class="description-box">
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.info(
            "Ürün açıklaması üretmek için "
            "OpenAI API anahtarı gerekiyor."
        )


   