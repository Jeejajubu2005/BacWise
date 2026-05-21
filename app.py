import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ==============================================================================
# 1. ตั้งค่าหน้าเว็บสไตล์แอปพลิเคชันทางการแพทย์ (ธีมสว่างดั้งเดิม)
# ==============================================================================
st.set_page_config(
    page_title="BacWise - Bacterial Classification",
    page_icon="🔬",
    layout="centered"
)

# 🛑 🛠️ ไม้ตายปิดบล็อก: ส่งแท็กคำสั่งบอกบราวเซอร์ว่า "ห้ามแปลภาษาหน้าเว็บนี้เด็ดขาด" 
# และปรับแต่งฟอนต์สารบรรณแบบคลีน ไม่ยุ่งกับปุ่ม เพื่อไม่ให้เกิดเลเยอร์หลอกตา
st.markdown("""
    <meta name="google" content="notranslate" />
    
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, button, label {
        font-family: 'Sarabun', sans-serif !important;
    }
    
    /* จัดแต่งข้อความหัวข้อหลัก */
    .main-title {
        font-weight: 700;
        color: #1E3A8A; 
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #4B5563;
        text-align: center;
        font-weight: 400;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัวข้อเว็บหลัก
st.markdown("<h1 class='main-title'>🔬 BacWise</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>ระบบปัญญาประดิษฐ์จำแนกสัณฐานและสีย้อมแกรมแบคทีเรียอัตโนมัติ</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# 2. ฟังก์ชันดาวน์โหลดและโหลดโมเดลด้วย gdown
# ==============================================================================
@st.cache_resource
def load_models():
    shape_id = '19xrNgp3STE2jx9odn-lAR1NSpMoFh2lm'
    color_id = '1XmLffNESJMIJ6kgtqLQlAXE5BTtf6r0w'
    
    shape_path = 'bacterial_shape_model_v3_balanced.h5'
    color_path = 'bacterial_color_model_v3_balanced.h5'
    
    if not os.path.exists(shape_path):
        url_shape = f'https://drive.google.com/uc?id={shape_id}'
        gdown.download(url_shape, shape_path, quiet=True)
        
    if not os.path.exists(color_path):
        url_color = f'https://drive.google.com/uc?id={color_id}'
        gdown.download(url_color, color_path, quiet=True)
        
    model_shape = tf.keras.models.load_model(shape_path)
    model_color = tf.keras.models.load_model(color_path)
    
    return model_shape, model_color

try:
    with st.spinner("🔄 ระบบกำลังโหลดโมเดลอัจฉริยะ..."):
        model_shape, model_color = load_models()
    st.success("✅ โหลดโมเดลระบบหลักเรียบร้อยแล้ว! พร้อมทำการวิเคราะห์")
except Exception as e:
    st.error("❌ ดึงข้อมูลล้มเหลว โปรดตรวจสอบสิทธิ์การแชร์ลิงก์ใน Google Drive")

shape_labels = ['Bacilli', 'Cocci', 'Spirals']

# ==============================================================================
# 3. ส่วนการรับภาพจากผู้ใช้งาน
# ==============================================================================
uploaded_file = st.file_uploader(
    "📸 อัปโหลดภาพถ่ายแบคทีเรียจากกล้องจุลทรรศน์ (.jpg, .jpeg, .png)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="🖼️ ภาพถ่ายแบคทีเรียที่นำเข้าสู่ระบบ", use_container_width=True)
    
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1,2,1])
    with col_btn_2:
        start_analysis = st.button("🔬 เริ่มทำการวิเคราะห์ทางจุลชีววิทยา", use_container_width=True)
        
    if start_analysis:
        img_resized = image_display.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # ==============================================================================
        # 4. ภาคประมวลผลร่วมกับระบบ Quality Control (Threshold ดักภาพไม่ชัด)
        # ==============================================================================
        THRESHOLD_SHAPE = 0.65  
        THRESHOLD_COLOR = 0.80  

        st.markdown("### 📊 ผลการวิเคราะห์จากระบบ")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### 🧬 สัณฐานวิทยา (Shape)")
            pred_shape = model_shape.predict(img_tensor)
            conf_shape = np.max(pred_shape[0])
            res_shape = shape_labels[np.argmax(pred_shape[0])]
            
            if conf_shape < THRESHOLD_SHAPE:
                st.warning("⚠️ **ข้อจำกัดในการวิเคราะห์**\n\nระบบไม่สามารถระบุรูปร่างได้อย่างมั่นใจ เนื่องจากขอบเซลล์เบลอ")
            else:
                st.metric(label="Bacterial Shape", value=res_shape)
                st.caption(f"ความมั่นใจของ AI: {conf_shape*100:.2f}%")
                
        with col_res2:
            st.markdown("#### 🎨 ผลสีย้อมแกรม (Gram)")
            pred_color = model_color.predict(img_tensor)
            raw_color_val = pred_color[0][0]
            
            if raw_color_val > 0.5:
                res_color = "Gram-Positive"
                res_sub = "ติดสีม่วง (Purple)"
                conf_color = raw_color_val
                color_theme = "#8A2BE2"  
            else:
                res_color = "Gram-Negative"
                res_sub = "ติดสีชมพู/แดง (Pink)"
                conf_color = 1 - raw_color_val
                color_theme = "#FF69B4"  
                
            if conf_color < THRESHOLD_COLOR:
                st.error("🎨 **ข้อจำกัดในการวิเคราะห์**\n\nระบบไม่สามารถระบุสีแกรมได้ เนื่องจากเฉดสีมีความเพี้ยน")
            else:
                st.markdown(f"<h2 style='color: {color_theme}; margin-bottom: 0px;'>{res_color}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #4B5563; font-size: 14px;'>{res_sub}</p>", unsafe_allow_html=True)
                st.caption(f"ความมั่นใจของ AI: {conf_color*100:.2f}%")

# ==============================================================================
# 5. ฟุตเตอร์ลิขสิทธิ์
# ==============================================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 12px;'>© 2026 BacWise Project. All Rights Reserved. (Private Repository Protected)</p>", unsafe_allow_html=True)
