import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ==============================================================================
# 1. ตั้งค่าหน้าเว็บสไตล์แอปพลิเคชันทางการแพทย์
# ==============================================================================
st.set_page_config(
    page_title="BacWise - Bacterial Classification",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 BacWise v3 (Balanced Dataset)")
st.subheader("ระบบปัญญาประดิษฐ์จำแนกสัณฐานและสีย้อมแกรมแบคทีเรีย")
st.markdown("---")

# ==============================================================================
# 2. ฟังก์ชันดาวน์โหลดและโหลดโมเดลด้วย gdown (ทะลวงบล็อกไฟล์ใหญ่)
# ==============================================================================
@st.cache_resource
def load_models():
    # ใช้วิธีการแกะ File ID จาก Google Drive ของคุณเหมือนเดิมเป๊ะครับ
    shape_id = '19xrNgp3STE2jx9odn-lAR1NSpMoFh2lm'
    color_id = '1XmLffNESJMIJ6kgtqLQlAXE5BTtf6r0w'
    
    # กำหนดชื่อไฟล์ปลายทางที่จะเซฟในเซิร์ฟเวอร์ Streamlit
    shape_path = 'bacterial_shape_model_v3_balanced.h5'
    color_path = 'bacterial_color_model_v3_balanced.h5'
    
    # 🚀 ตรวจสอบและดาวน์โหลดโมเดลรูปร่าง (ถ้ายังไม่มีในระบบ)
    if not os.path.exists(shape_path):
        url_shape = f'https://drive.google.com/uc?id={shape_id}'
        gdown.download(url_shape, shape_path, quiet=True)
        
    # 🚀 ตรวจสอบและดาวน์โหลดโมเดลสีแกรม (ถ้ายังไม่มีในระบบ)
    if not os.path.exists(color_path):
        url_color = f'https://drive.google.com/uc?id={color_id}'
        gdown.download(url_color, color_path, quiet=True)
        
    # โหลดโมเดลเข้าสู่ระบบ TensorFlow
    model_shape = tf.keras.models.load_model(shape_path)
    model_color = tf.keras.models.load_model(color_path)
    
    return model_shape, model_color

# พยายามโหลดโมเดลและแจ้งสถานะบนหน้าเว็บ
try:
    with st.spinner("🔄 ระบบกำลังใช้พลังทะลวงดาวน์โหลดโมเดลขนาดใหญ่จาก Google Drive (ครั้งแรกอาจใช้เวลา 1-2 นาที)..."):
        model_shape, model_color = load_models()
    st.success("✅ โหลดโมเดล BacWise v3 ผ่านการดึงไฟล์ใหญ่เรียบร้อยแล้ว! ระบบพร้อมวิเคราะห์")
except Exception as e:
    st.error("❌ ดึงข้อมูลล้มเหลว โปรดตรวจสอบสิทธิ์การแชร์ลิงก์ใน Google Drive ให้เป็น 'Anyone with the link' หรือลองกดรันแอปใหม่อีกครั้ง")

# รายชื่อคลาสสัณฐานวิทยา (ตามลำดับการเทรน)
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
    
    if st.button("🔬 เริ่มทำการวิเคราะห์ทางจุลชีววิทยา"):
        
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
        
        # 🟢 4.1 วิเคราะห์ฝั่งรูปร่าง (Shape Model)
        pred_shape = model_shape.predict(img_tensor)
        conf_shape = np.max(pred_shape[0])
        res_shape = shape_labels[np.argmax(pred_shape[0])]
        
        if conf_shape < THRESHOLD_SHAPE:
            st.warning("⚠️ **ข้อจำกัดในการวิเคราะห์โครงสร้างเซลล์**")
            st.info("ระบบไม่สามารถระบุรูปร่างได้อย่างมั่นใจ เนื่องจากความคมชัดของภาพต่ำหรือขอบเซลล์เบลอ \n\n*คำแนะนำ: โปรดอัปโหลดภาพที่มีระยะโฟกัสที่นิ่งและเห็นขอบเซลล์แบคทีเรียชัดเจน*")
        else:
            st.metric(label="🧬 ผลลัพธ์สัณฐานวิทยา (Bacterial Shape)", value=res_shape)
            st.caption(f"ดัชนีความมั่นใจของ AI: {conf_shape*100:.2f}%")
            
        st.markdown("---")
        
        # 🔵 4.2 วิเคราะห์ฝั่งสีแกรม (Color Model)
        pred_color = model_color.predict(img_tensor)
        raw_color_val = pred_color[0][0]
        
        if raw_color_val > 0.5:
            res_color = "Gram-Positive Bacteria (ติดสีม่วง)"
            conf_color = raw_color_val
            color_theme = "#8A2BE2"  
        else:
            res_color = "Gram-Negative Bacteria (ติดสีชมพู/แดง)"
            conf_color = 1 - raw_color_val
            color_theme = "#FF69B4"  
            
        if conf_color < THRESHOLD_COLOR:
            st.error("🎨 **ข้อจำกัดในการจำแนกประเภทสีย้อมแกรม**")
            st.info("ระบบไม่สามารถระบุสีแกรมได้ เนื่องจากความเข้มสีต่ำกว่าเกณฑ์หรือเฉดสีมีความเพี้ยน \n\n*คำแนะนำ: โปรดตรวจสอบเทคนิคการย้อมเชื้อ หรือปรับค่าสมดุลแสง (White Balance) ของกล้องจุลทรรศน์ให้เห็นเฉดสีม่วงหรือชมพูที่แท้จริง*")
        else:
            st.markdown("##### การย้อมติดสีแกรม (Gram Staining)")
            st.markdown(f"<h2 style='color: {color_theme};'>{res_color}</h2>", unsafe_allow_html=True)
            st.caption(f"ดัชนีความมั่นใจของ AI: {conf_color*100:.2f}%")

# ==============================================================================
# 5. ฟุตเตอร์ลิขสิทธิ์
# ==============================================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 BacWise Project. All Rights Reserved. (Private Repository Protected)</p>", unsafe_allow_html=True)
