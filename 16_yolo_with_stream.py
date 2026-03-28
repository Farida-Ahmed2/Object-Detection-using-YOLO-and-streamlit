import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Yolo Project",
    page_icon=":bar_chart:", 
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit as st
# ↑ Streamlit — لبناء الواجهة

from PIL import Image
# ↑ PIL (Pillow) — مكتبة لمعالجة الصور
# فتح الصور / تحويل الصيغ / تغيير الحجم
# Pillow هي النسخة الحديثة من PIL (Python Imaging Library)

import numpy as np
# ↑ NumPy — مكتبة للعمليات الرياضية على المصفوفات
# الصور في الكمبيوتر = مصفوفات أرقام (كل رقم = لون بكسل)
# YOLOv8 يتعامل مع الصور كـ numpy arrays

import io
# ↑ io — للتعامل مع تدفقات البيانات (streams)
# نستخدمه لتحويل الصورة بين الصيغ (bytes ↔ Image)


# ─── إعداد الصفحة ────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection - YOLOv8",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS مخصص ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══ إخفاء شريط Toolbar ═══ */
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }

/* ═══ دعم RTL للعربية ═══ */
.stApp { direction: rtl !important; }
h1, h2, h3, h4, h5, h6, p, div, label, span { text-align: right !important; }
input, textarea { direction: rtl !important; text-align: right !important; }
[data-baseweb="slider"] { direction: ltr; }
section[data-testid="stSidebar"] { direction: rtl; }

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

:root {
    --glass-bg: rgba(15, 15, 40, 0.7);
    --glass-border: rgba(124, 58, 237, 0.15);
}

.stApp {
    background: linear-gradient(160deg, #0a0a1a 0%, #12122a 30%, #0d1025 60%, #0a0a1a 100%) !important;
    font-family: 'Inter', 'Tajawal', sans-serif !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12,12,35,0.97) 0%, rgba(8,8,25,0.99) 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

.main-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #06b6d4, #7c3aed);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite;
    text-align: center; margin-bottom: 0.3rem;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}
.sub-title { color: #94a3b8; text-align: center; font-size: 1.1rem; margin-bottom: 2rem; }
.glass-card {
    background: var(--glass-bg); border: 1px solid var(--glass-border);
    border-radius: 20px; padding: 1.8rem; backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124,58,237,0.35); transform: translateY(-2px);
}
.premium-divider {
    height: 1px; background: linear-gradient(90deg, transparent, rgba(124,58,237,0.4), transparent);
    margin: 1.5rem 0; border: none;
}
.stat-card {
    background: var(--glass-bg); border: 1px solid var(--glass-border);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    backdrop-filter: blur(12px); transition: all 0.3s ease;
}
.stat-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(124,58,237,0.15); }
.stat-number {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { color: #94a3b8; font-size: 0.9rem; margin-top: 0.4rem; font-weight: 500; }
.badge {
    display: inline-block; background: linear-gradient(135deg, #7c3aed, #06b6d4);
    color: white; padding: 0.25rem 0.8rem; border-radius: 100px;
    font-size: 0.75rem; font-weight: 600;
}

/* ═══ تصميم نتائج الكشف ═══ */
.detection-result {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 14px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    display: flex;
    justify-content: space-between;
    /* ↑ يوزع المحتوى بين اليمين واليسار
       الاسم على اليسار والنسبة على اليمين */
    align-items: center;
    color: #f1f5f9;
    transition: all 0.2s ease;
}
.detection-result:hover {
    background: rgba(6,182,212,0.15);
    transform: translateX(4px);
    /* ↑ ينزاح يميناً 4px عند hover — حركة خفيفة وأنيقة */
}
.detection-label { font-weight: 600; font-size: 1rem; }
.detection-conf {
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 700; font-size: 1.05rem;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,0.45) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1️⃣ Session State — عداد الكشوفات
# ══════════════════════════════════════════════════════════════
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0


# ══════════════════════════════════════════════════════════════
# 2️⃣ الشريط الجانبي — الإعدادات
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.6rem;">📷 YOLOv8</div>', unsafe_allow_html=True)
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ── إحصائيات ──
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.total_detections}</div>
        <div class="stat-label">🎯 أجسام مكتشفة</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # 📝 إعدادات الكشف
    # ══════════════════════════════════════════════════════════

    st.markdown("#### ⚙️ إعدادات الكشف")

    conf_threshold = st.slider(
        "🎯 حد الثقة (Confidence Threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    # ↑ Confidence Threshold = أقل نسبة ثقة لقبول الكشف
    #
    # 📝 شرح مفهوم Confidence:
    # ─────────────────────────────────────────────────────────
    # كل ما YOLO يكتشف جسم، يعطيه نسبة ثقة (0.0 - 1.0)
    # مثال: كشف سيارة بثقة 0.92 = 92% متأكد إنها سيارة
    #
    # - threshold = 0.25 → يعرض كل شيء (حتى اللي مو متأكد منه)
    # - threshold = 0.50 → يعرض الأجسام المتأكد منها نسبياً
    # - threshold = 0.80 → يعرض فقط الأجسام المتأكد منها جداً
    #
    # ⚠️ أفضل ممارسة:
    # - 0.25 = للاستكشاف (معرفة كل شيء ممكن)
    # - 0.50 = للاستخدام العام
    # - 0.70+ = للإنتاج (دقة عالية، أخطاء أقل)

    st.caption(f"القيمة الحالية: {conf_threshold:.0%}")
    # ↑ st.caption = نص صغير رمادي
    # :.0% = تنسيق النسبة بدون فاصلة عشرية (مثلاً 25%)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    model_size = st.selectbox(
        "📦 حجم الموديل",
        ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        index=0,
    )
    # ↑ YOLOv8 يأتي بـ 5 أحجام مختلفة:
    #
    # ┌──────────┬──────────┬──────────┬──────────────┐
    # │ الموديل  │ الحجم    │ السرعة   │ الدقة        │
    # ├──────────┼──────────┼──────────┼──────────────┤
    # │ yolov8n  │ 6.2 MB   │ أسرع 🚀 │ أقل دقة     │
    # │ yolov8s  │ 21.5 MB  │ سريع    │ جيدة         │
    # │ yolov8m  │ 49.7 MB  │ متوسط   │ جيدة جداً    │
    # │ yolov8l  │ 83.7 MB  │ بطيء    │ عالية        │
    # │ yolov8x  │ 130.5 MB │ أبطأ 🐢 │ أعلى دقة ⭐  │
    # └──────────┴──────────┴──────────┴──────────────┘
    #
    # n = nano (صغير وسريع) — ممتاز للتعلم والتجربة
    # x = extra large (كبير ودقيق) — للإنتاج
    #
    # ⚠️ أفضل ممارسة: ابدأ بـ nano للتعلم
    # لما تحتاج دقة أعلى، جرب s أو m

    # شرح الأحجام
    size_info = {
        "yolov8n": "🚀 Nano — سريع جداً (~6MB)",
        "yolov8s": "⚡ Small — متوازن (~22MB)",
        "yolov8m": "📊 Medium — دقة أعلى (~50MB)",
        "yolov8l": "💪 Large — دقة عالية (~84MB)",
        "yolov8x": "🎯 Extra Large — أعلى دقة (~131MB)",
    }
    st.caption(size_info[model_size])

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # معلومات عن YOLO
    st.markdown("""
    <div style="padding:0.8rem; background:rgba(15,15,40,0.6); border-radius:12px; border:1px solid rgba(124,58,237,0.15);">
        <div style="color:#94a3b8; font-size:0.75rem; margin-bottom:0.4rem;">ما هو YOLO؟</div>
        <div style="color:#f1f5f9; font-size:0.8rem; line-height:1.6;">
            <strong>Y</strong>ou <strong>O</strong>nly <strong>L</strong>ook <strong>O</strong>nce<br>
            خوارزمية كشف أجسام بالوقت الحقيقي.<br>
            تكتشف 80+ نوع من الأجسام.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 3️⃣ المحتوى الرئيسي
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">📷 كشف الأجسام — YOLOv8</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ارفع صورة وشاهد YOLOv8 يكتشف الأجسام فيها</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 4️⃣ رفع الصورة — File Uploader
# ══════════════════════════════════════════════════════════════
# st.file_uploader يعرض منطقة رفع ملفات (drag & drop + زر)
# نحدد الأنواع المسموحة لمنع رفع ملفات غير مدعومة
# ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📤 ارفع صورة للتحليل",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    # ↑ نحدد امتدادات الصور المسموحة فقط
    # ⚠️ أفضل ممارسة: دائماً حدد الأنواع المسموحة
    # لمنع المستخدم من رفع ملفات غير مدعومة
    label_visibility="collapsed",
    # ↑ نخفي التسمية — لأن التصميم يوضح الغرض
)

if uploaded_file is not None:
    # ↑ لما المستخدم يرفع صورة

    # ══════════════════════════════════════════════════════════
    # 5️⃣ فتح الصورة بـ Pillow
    # ══════════════════════════════════════════════════════════
    image = Image.open(uploaded_file)
    # ↑ Image.open يفتح الصورة من الملف المرفوع
    # يرجع كائن Image من Pillow يحتوي بيانات الصورة

    # عرض الصورة الأصلية والنتيجة جنب بعض
    col_orig, col_result = st.columns(2)
    # ↑ عمودين متساويين: الأصلية على اليسار، النتيجة على اليمين

    with col_orig:
        st.markdown("### 🖼️ الصورة الأصلية")
        st.image(image, width="stretch")
        # ↑ st.image يعرض الصورة
        # use_container_width=True = الصورة تأخذ كامل عرض العمود
        # هذا أفضل من تحديد عرض ثابت — يتجاوب مع حجم الشاشة

    # ══════════════════════════════════════════════════════════
    # 6️⃣ تحميل وتشغيل YOLOv8
    # ══════════════════════════════════════════════════════════
    with st.spinner("🔍 جاري تحليل الصورة بـ YOLOv8..."):
        # ↑ نعرض أنيميشن تحميل أثناء التحليل

        try:
            from ultralytics import YOLO
            # ↑ نستورد YOLO من مكتبة ultralytics
            # ⚠️ نستورد هنا وليس في أعلى الملف لأن:
            # 1. المكتبة كبيرة — الاستيراد ياخذ وقت
            # 2. لو المكتبة مو مثبتة، نقدر نعرض رسالة خطأ مفيدة
            # 3. ما نبطئ تحميل الصفحة إلا لما المستخدم يحتاجها فعلاً

            # ═══════════════════════════════════════════════════
            # 💡 st.cache_resource — التخزين المؤقت للموارد الثقيلة
            # ═══════════════════════════════════════════════════
            # بدون cache: كل ما المستخدم يرفع صورة جديدة
            # Streamlit يعيد تشغيل الكود → يحمّل الموديل من جديد!
            #
            # مع cache: الموديل يتحمّل مرة وحدة فقط
            # كل المرات الجاية يستخدم النسخة المحفوظة → أسرع بكثير
            #
            # @st.cache_resource = للموارد الثقيلة (موديلات ML, اتصالات DB)
            # @st.cache_data = للبيانات (DataFrames, نتائج API)
            #
            # ⚠️ أفضل ممارسة: دائماً استخدم cache للموديلات
            # تحميل موديل كل مرة = هدر وقت + ذاكرة
            # ═══════════════════════════════════════════════════

            @st.cache_resource
            def load_yolo_model(size):
                """
                تحميل موديل YOLOv8 مع تخزين مؤقت.
                
                المعاملات:
                    size: حجم الموديل ("yolov8n", "yolov8s", إلخ)
                
                المرجع:
                    كائن YOLO جاهز للاستخدام
                
                ملاحظة: أول تشغيل سيحمّل الموديل من الإنترنت
                المرات التالية سيستخدم النسخة المحفوظة
                """
                return YOLO(f"{size}.pt")
                # ↑ YOLO("yolov8n.pt") يحمّل الموديل
                # .pt = صيغة PyTorch (إطار عمل تعلم عميق)
                # لو الملف مو موجود → يحمّله تلقائياً من الإنترنت

            model = load_yolo_model(model_size)
            # ↑ نحمّل الموديل (أو نستخدم النسخة المحفوظة)

            # ═══════════════════════════════════════════════════
            # 7️⃣ تحويل الصورة وتشغيل الكشف
            # ═══════════════════════════════════════════════════

            img_array = np.array(image)
            # ↑ نحول الصورة من كائن Pillow إلى مصفوفة NumPy
            # المصفوفة شكلها: (height, width, 3)
            # 3 = ثلاث قنوات لون: Red, Green, Blue (RGB)
            # كل قيمة = رقم من 0 (أسود) إلى 255 (أبيض/ألمع)
            #
            # مثال: صورة 640×480 = مصفوفة بشكل (480, 640, 3)
            # يعني 480 صف × 640 عمود × 3 ألوان = 921,600 رقم!

            results = model(img_array, conf=conf_threshold)
            # ↑ نشغّل الموديل على الصورة
            # conf = حد الثقة الأدنى (اللي حدده المستخدم)
            # results = قائمة نتائج (عادة عنصر واحد لصورة واحدة)

            result = results[0]
            # ↑ نأخذ نتيجة أول صورة (وحيدة عندنا)
            # result يحتوي:
            # - result.boxes = كل الأجسام المكتشفة (مواقعها وأنواعها)
            # - result.names = أسماء الفئات {0: "person", 1: "bicycle", ...}
            # - result.plot() = يرسم المربعات على الصورة

            # ═══════════════════════════════════════════════════
            # 8️⃣ رسم المربعات على الصورة (Annotated Image)
            # ═══════════════════════════════════════════════════

            annotated = result.plot()
            # ↑ يرسم مربعات ملونة حول كل جسم مكتشف
            # مع اسم الفئة ونسبة الثقة
            # يرجع صورة بصيغة BGR (مو RGB!) ← لأن OpenCV يستخدم BGR

            annotated_image = Image.fromarray(annotated[..., ::-1])
            # ↑ نحول BGR → RGB:
            # annotated[..., ::-1] = نعكس ترتيب قنوات اللون
            # BGR = Blue, Green, Red → RGB = Red, Green, Blue
            # [..., ::-1] = آخر بُعد (اللون) نعكسه
            #
            # ثم Image.fromarray = نحول من numpy array إلى Pillow Image
            # عشان نقدر نعرضها بـ st.image

            with col_result:
                st.markdown("### 🎯 نتائج الكشف")
                st.image(annotated_image, width="stretch")

            st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

            # ═══════════════════════════════════════════════════
            # 9️⃣ عرض تفاصيل النتائج
            # ═══════════════════════════════════════════════════

            boxes = result.boxes
            # ↑ result.boxes يحتوي كل الأجسام المكتشفة
            # كل box فيه:
            # - box.cls = رقم الفئة (0 = person, 1 = bicycle, إلخ)
            # - box.conf = نسبة الثقة (0.0 - 1.0)
            # - box.xyxy = إحداثيات المربع (x1, y1, x2, y2)

            if len(boxes) > 0:
                # ↑ لو في أجسام مكتشفة

                st.session_state.total_detections += len(boxes)
                # ↑ نحدّث العداد الإجمالي

                st.markdown("### 📊 الأجسام المكتشفة")

                # ── جمع معلومات الفئات ──
                class_names = [result.names[int(cls)] for cls in boxes.cls]
                # ↑ نحول أرقام الفئات إلى أسماء
                # boxes.cls = [0, 0, 2, 7] (أرقام)
                # result.names = {0: "person", 2: "car", 7: "truck"}
                # class_names = ["person", "person", "car", "truck"]

                unique_classes = list(set(class_names))
                # ↑ set = يشيل التكرار → نحصل على الأنواع الفريدة فقط
                # مثال: ["person", "person", "car"] → {"person", "car"}
                # list = نحولها مرة ثانية لقائمة

                # ── بطاقات إحصائية ──
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{len(boxes)}</div>
                        <div class="stat-label">إجمالي الأجسام</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stat_cols[1]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{len(unique_classes)}</div>
                        <div class="stat-label">أنواع فريدة</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stat_cols[2]:
                    avg_conf = float(boxes.conf.mean()) * 100
                    # ↑ boxes.conf = كل نسب الثقة (tensor)
                    # .mean() = المتوسط الحسابي
                    # * 100 = تحويل من 0-1 إلى 0-100%
                    # float() = تحويل من tensor إلى رقم عادي
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{avg_conf:.1f}%</div>
                        <div class="stat-label">متوسط الثقة</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

                # ── تفاصيل كل جسم مكتشف ──
                st.markdown("### 📋 التفاصيل")

                for i, box in enumerate(boxes):
                    # ↑ enumerate يعطينا الرقم التسلسلي (i) مع العنصر (box)
                    cls_name = result.names[int(box.cls)]
                    # ↑ نجيب اسم الفئة من رقمها
                    conf = float(box.conf) * 100
                    # ↑ نسبة الثقة كنسبة مئوية

                    st.markdown(f"""
                    <div class="detection-result">
                        <span class="detection-label">🏷️ {cls_name}</span>
                        <span class="detection-conf">{conf:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # ↑ ما لقى أي أجسام
                st.markdown("""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">🔍</div>
                    <h3 style="color:#f1f5f9;">ما تم اكتشاف أي أجسام</h3>
                    <p style="color:#94a3b8;">جرب تقلل حد الثقة أو ارفع صورة مختلفة</p>
                </div>
                """, unsafe_allow_html=True)

        except ImportError:
            # ↑ المكتبة مو مثبتة
            st.error("❌ مكتبة ultralytics مو مثبتة!")
            st.code("pip install ultralytics", language="bash")

        except Exception as e:
            # ↑ أي خطأ ثاني
            st.error(f"❌ خطأ في التحليل: {str(e)}")

else:
    # ↑ ما رفع صورة بعد — نعرض placeholder جميل
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:3rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📷</div>
        <h3 style="color:#f1f5f9; margin-bottom:0.5rem;">ارفع صورة للتحليل</h3>
        <p style="color:#94a3b8;">اسحب وأفلت أو اضغط لرفع صورة</p>
        <div style="margin-top:1rem;">
            <span class="badge">JPG</span>&nbsp;
            <span class="badge">PNG</span>&nbsp;
            <span class="badge">BMP</span>&nbsp;
            <span class="badge">WEBP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
