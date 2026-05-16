from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import os
import cv2
import matplotlib.cm as cm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from reportlab.pdfgen import canvas
from datetime import datetime
import csv
from flask import Response
from flask import jsonify
from datetime import timedelta

from reportlab.platypus import PageBreak
# from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as ReportImage
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "pneumodetect_secret"

# ================= PATH CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "pneumo_clean_model.h5")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

def build_gradcam_model(original_model):

    base_model = original_model.get_layer('mobilenetv2_1.00_224')

    x = base_model.output
    x = original_model.layers[1](x)
    x = original_model.layers[2](x)
    output = original_model.layers[3](x)

    functional_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=output
    )

    return functional_model

gradcam_model = build_gradcam_model(model)

base_model = model.layers[0]
class_names = ['BACTERIAL', 'NORMAL', 'VIRAL']


def build_gradcam_model(original_model):

    # Extract base mobilenet model
    base_model = original_model.get_layer('mobilenetv2_1.00_224')

    # Rebuild full functional graph properly
    x = base_model.output
    x = original_model.layers[1](x)   # GlobalAveragePooling
    x = original_model.layers[2](x)   # Dropout
    output = original_model.layers[3](x)  # Dense

    functional_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=output
    )

    return functional_model

# ================= DATABASE =================
def init_db():
    print("DATABASE FILE PATH:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT,
        email TEXT UNIQUE,
        contact TEXT,
        specialization TEXT,
        hospital TEXT,
        password TEXT,
        is_active INTEGER DEFAULT 1
        )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_email TEXT,
        patient_name TEXT,
        age INTEGER,
        gender TEXT,
        prediction TEXT,
        confidence REAL,
        image_path TEXT,
        gradcam_path TEXT,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    role TEXT,
    action TEXT,
    details TEXT,
    timestamp DATETIME DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.commit()
    conn.close()

init_db()


def add_log(user, role, action, details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO system_logs (user_name, role, action, details)
        VALUES (?, ?, ?, ?)
    """, (user, role, action, details))

    conn.commit()
    conn.close()

# ================= GRAD-CAM =================
def make_gradcam_heatmap(img_array, model):

    # Get last conv layer from MobileNet
    last_conv_layer = model.get_layer('Conv_1')

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def is_valid_xray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    # Check if image is mostly grayscale
    b, g, r = cv2.split(img)

    diff_rg = np.mean(np.abs(r - g))
    diff_gb = np.mean(np.abs(g - b))
    diff_rb = np.mean(np.abs(r - b))

    avg_diff = (diff_rg + diff_gb + diff_rb) / 3

    # X-ray images are nearly grayscale
    if avg_diff > 20:
        return False

    # Check image size (avoid tiny random images)
    h, w, _ = img.shape
    if h < 200 or w < 200:
        return False

    return True

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form["fullname"]
        email = request.form["email"]
        contact = request.form["contact"]
        specialization = request.form["specialization"]
        hospital = request.form["hospital"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (fullname, email, contact, specialization, hospital, password)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (fullname, email, contact, specialization, hospital, hashed_password))
            conn.commit()
            conn.close()

            add_log(fullname, "Doctor", "Register", "New doctor registered")

            flash("Registration Successful! Please login.")
            return redirect(url_for("login"))
        except:
            flash("Email already exists!")
            return redirect(url_for("register"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[6], password):

            if user[7] == 0:
                flash("Account is deactivated by admin.")
                return redirect(url_for("login"))

            session["user"] = user[1]
            session["email"] = user[2]

            print("LOGIN SUCCESS LOG TRIGGERED")
            add_log(user[1], "Doctor", "Login", "Doctor logged into system")

            return redirect(url_for("doctor_dashboard"))

        else:
            print("FAILED LOGIN LOG TRIGGERED")
            add_log(email, "Doctor", "Failed Login", "Invalid email or password")
            flash("Invalid Email or Password!")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/doctor-dashboard")
def doctor_dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("doctor_dashboard.html")

@app.route("/dashboard-stats")
def dashboard_stats():

    if "user" not in session:
        return {"error": "Unauthorized"}, 401

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM reports WHERE doctor_email=?",
                   (session["email"],))
    total_reports = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE doctor_email=? AND prediction='BACTERIAL'
    """, (session["email"],))
    bacterial = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE doctor_email=? AND prediction='VIRAL'
    """, (session["email"],))
    viral = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE doctor_email=? AND prediction='NORMAL'
    """, (session["email"],))
    normal = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE doctor_email=? AND date(date)=date('now')
    """, (session["email"],))
    today = cursor.fetchone()[0]

    conn.close()

    return {
        "totalPatients": total_reports,   # same as reports for now
        "totalReports": total_reports,
        "bacterialCases": bacterial,
        "viralCases": viral,
        "normalCases": normal,
        "todayReports": today
    }


@app.route("/upload")
def upload_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("upload_xray.html")

@app.route("/reports")
def reports_page():

    if "user" not in session:
        return redirect(url_for("login"))

    search_query = request.args.get("search")
    filter_type = request.args.get("filter")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT id, patient_name, age, gender, prediction, confidence, date
        FROM reports
        WHERE doctor_email=?
    """

    params = [session["email"]]

    # 🔎 Search Filter
    if search_query:
        query += " AND patient_name LIKE ?"
        params.append(f"%{search_query}%")

    # 📊 Dashboard Card Filter
    if filter_type == "bacterial":
        query += " AND prediction='BACTERIAL'"
    elif filter_type == "viral":
        query += " AND prediction='VIRAL'"
    elif filter_type == "normal":
        query += " AND prediction='NORMAL'"
    elif filter_type == "today":
        query += " AND date(date)=date('now')"
    # filter=all → no extra condition

    query += " ORDER BY date DESC"

    cursor.execute(query, params)
    reports = cursor.fetchall()

    conn.close()

    return render_template("reports.html", reports=reports)


@app.route("/delete-report/<int:report_id>")
def delete_report(report_id):

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM reports
        WHERE id=? AND doctor_email=?
    """, (report_id, session["email"]))

    conn.commit()
    conn.close()

    add_log(session["user"], "Doctor", "Delete Report",
            f"Deleted report ID {report_id}")

    return redirect(url_for("reports_page"))

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    patient_name = request.form["patient_name"]
    age = request.form["age"]
    gender = request.form["gender"]

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # -------- VALIDATION --------
    if not is_valid_xray(filepath):
        flash("Invalid Image! Please upload a valid Chest X-ray image.")
        return redirect(url_for("upload_page"))

    original_pil = PILImage.open(filepath).convert("RGB")

    if original_pil.size[0] < 150 or original_pil.size[1] < 150:
        flash("Image resolution too low!")
        return redirect(url_for("upload_page"))

    img = np.array(original_pil)

    img = cv2.GaussianBlur(img, (3,3), 0)

    model_img = cv2.resize(img, (224,224))
    img_array = np.expand_dims(model_img, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model(img_array, training=False)
    probabilities = predictions[0]

    # ✅ FINAL PREDICTION
    confidence = float(np.max(probabilities) * 100)
    predicted_class = class_names[np.argmax(probabilities)]

    # ✅ MESSAGE
    if confidence > 85:
        message = "High Confidence Result"
    elif confidence > 60:
        message = "Moderate Confidence - Doctor Consultation Recommended"
    else:
        message = "Low Confidence - Please Re-test or Consult Doctor"

    # ✅ EXPLANATION
    if predicted_class == "BACTERIAL":
        explanation = "Possible lung opacity and localized infection detected."
    elif predicted_class == "VIRAL":
        explanation = "Diffuse infection pattern observed in lungs."
    else:
        explanation = "No visible signs of pneumonia."

    # -------- GRADCAM --------
    heatmap = make_gradcam_heatmap(img_array, gradcam_model)

    original_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_color, 0.3, 0)

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.75, heatmap_color, 0.35, 0)

    gradcam_filename = "gradcam_" + file.filename
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, superimposed_img)

    # ✅ VARIABLES
    image_name = file.filename
    heatmap_url = url_for('static', filename='uploads/' + gradcam_filename)

    # -------- SAVE TO DB -------- ✅ (RETURN chya aadhi!)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reports 
        (
        doctor_email,
        patient_name,
        age,
        gender,
        prediction,
        confidence,
        image_path,
        gradcam_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["email"],
        patient_name,
        age,
        gender,
        predicted_class,
        round(confidence, 2),
        filepath,
        gradcam_path
    ))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()

    add_log(session["user"], "Doctor", "Prediction",
            f"Predicted {predicted_class} for patient {patient_name}")

    # ✅ FINAL RETURN (ONLY ONCE 🔥)
    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        message=message,
        explanation=explanation,
        heatmap_url=heatmap_url,
        image_name=image_name,
        report_id=report_id,
        patient_name=patient_name,
        age=age,
        gender=gender
    )

def add_page_number(canvas, doc):

    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"

    canvas.setFont("Helvetica", 9)

    canvas.drawRightString(
        570,
        20,
        text
    )

    canvas.drawString(
        40,
        20,
        "PneumoDetect System"
    )

@app.route("/download-report/<int:report_id>")
def download_report(report_id):

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT patient_name, age, gender, prediction, confidence, image_path, gradcam_path
        FROM reports
        WHERE id=? AND doctor_email=?
    """, (report_id, session["email"]))

    report = cursor.fetchone()
    conn.close()

    if not report:
        return "Report not found"

    patient, age, gender, prediction, confidence, filepath, gradcam_path = report

    pdf_path = os.path.join(UPLOAD_FOLDER, f"report_{report_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, topMargin=30, bottomMargin=30)

  
    elements = []
    styles = getSampleStyleSheet()

    # -------- STYLES --------
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=23,
        alignment=1,
        textColor=colors.HexColor("#0B5E5E"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        alignment=1,
        fontSize=11,
        textColor=colors.grey,
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor("#0B3D91"),
        spaceAfter=4
    )

    normal_style = styles["Normal"]

    # -------- TITLE --------
    elements.append(Paragraph("REPORT", title_style))
    
    elements.append(Paragraph("AI Generated Diagnostic Report", subtitle_style))

    # elements.append(HRFlowable(width="100%", thickness=3,
    #                         color=colors.HexColor("#A7D8F0")))
    # elements.append(Spacer(1, 0.2 * inch))

    # -------- REPORT INFORMATION --------
    elements.append(Paragraph("Information", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                            color=colors.lightgrey))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph(f"<b>Report ID:</b> {report_id}", normal_style))
    elements.append(Paragraph(f"<b>Consultant Radiologist:</b> Dr. Alexander Smith", normal_style))
    elements.append(Paragraph(f"<b>Contact No:</b> 945067XXXX", normal_style))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y')}", normal_style))
    elements.append(Paragraph(f"<b>Time:</b> {datetime.now().strftime('%H:%M')}", normal_style))

    elements.append(Spacer(1, 0.2 * inch))

    # -------- EXAMINATION DETAILS --------
    elements.append(Paragraph("Examination Details", section_style))

    elements.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.lightgrey
    ))

    elements.append(Spacer(1, 0.12 * inch))

    # Examination Text
    elements.append(Paragraph("Chest X-Ray (PA View)", normal_style))
    elements.append(Paragraph("Technique: Digital Radiography", normal_style))

    elements.append(Spacer(1, 0.18 * inch))

    # X-Ray Image
    xray_img = ReportImage(filepath, width=220, height=220)
    xray_img.hAlign = 'CENTER'

    elements.append(xray_img)

    # Image Name Below Image
    elements.append(Spacer(1, 0.08 * inch))

    image_name_style = ParagraphStyle(
        'imgname',
        parent=styles['Normal'],
        alignment=1,   # CENTER
        fontSize=10,
        textColor=colors.grey
    )

    elements.append(
        Paragraph(
            f"Original Chest X-Ray Image",
            image_name_style
        )
    )

    elements.append(Spacer(1, 0.2 * inch))

   # -------- PATIENT INFO --------
    elements.append(Paragraph("Patient Information", section_style))

    elements.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.lightgrey
    ))

    elements.append(Spacer(1, 0.1 * inch))

    patient_table = Table([
        ["Name", patient],
        ["Age", age],
        ["Gender", gender]
    ], colWidths=[90, 360])

    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
    ]))

    elements.append(patient_table)

    elements.append(Spacer(1, 0.2 * inch))


    # -------- FINDINGS --------
    elements.append(Paragraph("Findings", section_style))

    elements.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.lightgrey
    ))

    elements.append(Spacer(1, 0.12 * inch))

    # Findings Text
    if prediction == "NORMAL":
        findings_text = """
        The chest X-ray image shows clear lung fields without significant white patchy opacities or dense consolidations.
        The AI model focused mainly on both lung regions, airway visibility, and overall opacity distribution while analyzing the image.
        Based on the observed lung texture patterns and absence of abnormal infected areas, the model prediction indicates NORMAL findings.
        """
    else:
        findings_text = """
        The chest X-ray image shows visible white patchy opacities and consolidation areas mainly in the lower and central lung regions.
        The AI model focused on abnormal bright regions, infected lung tissues, and uneven opacity distribution during image analysis.
        Based on these detected radiological patterns and highlighted affected areas, the model prediction suggests PNEUMONIA findings.
        """

    elements.append(Paragraph(findings_text, normal_style))

    elements.append(Spacer(1, 0.2 * inch))

    # IMAGE ON NEXT PAGE
    elements.append(PageBreak())

    # -------- HEATMAP IMAGE --------
    gradcam_img = ReportImage(gradcam_path, width=220, height=220)
    gradcam_img.hAlign = 'CENTER'

    elements.append(gradcam_img)

    # IMAGE NAME
    elements.append(Spacer(1, 0.08 * inch))

    image_name_style = ParagraphStyle(
        'imgname',
        parent=styles['Normal'],
        alignment=1,
        fontSize=10,
        textColor=colors.grey
    )

    elements.append(
        Paragraph(
            f"Grad-CAM Heatmap",
            image_name_style
        )
    )

    elements.append(Spacer(1, 0.25 * inch))

    # -------- AI ANALYSIS --------
    elements.append(Paragraph("AI Model Analysis", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                            color=colors.lightgrey))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph(f"<b>Prediction:</b> {prediction} PNEUMONIA", normal_style))
    elements.append(Paragraph(f"<b>Confidence Score:</b> {confidence}%", normal_style))
    elements.append(Paragraph("<b>Model:</b> CNN (MobileNetV2 Based)", normal_style))

    elements.append(Spacer(1, 0.2 * inch))

    # -------- IMPRESSION --------
    elements.append(Paragraph("Impression", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                            color=colors.lightgrey))
    elements.append(Spacer(1, 0.1 * inch))

    if prediction == "NORMAL":
        impression = "No radiographic evidence of Pneumonia."
    else:
        impression = "Radiographic findings suggest Pneumonia. Clinical correlation recommended."

    elements.append(Paragraph(impression, normal_style))

    elements.append(Spacer(1, 0.25 * inch))

    # -------- WARNING --------
    elements.append(Paragraph(
        "⚠ This report is generated using an AI-based diagnostic support system. "
        "Final diagnosis must be confirmed by a qualified medical professional.",
        normal_style
    ))

    elements.append(Spacer(1, 0.35 * inch))

    # -------- SIGNATURE RIGHT --------
    signature_table = Table(
        [["Signature: ___________________________"]],
        colWidths=[450]
    )

    signature_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT')
    ]))

    elements.append(signature_table)
    elements.append(Spacer(1, 0.1 * inch))
    # elements.append(Paragraph(" Dr. Alexander Smith", normal_style))

    dr_style = ParagraphStyle(
        'drstyle',
        parent=styles['Normal'],
        alignment=2
    )

    elements.append(Paragraph(" Dr. Alexander Smith", dr_style))


    doc.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )
    return send_file(pdf_path, as_attachment=True)


# ---------- PROFILE ----------
@app.route("/profile", methods=["GET", "POST"])
def profile_page():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if request.method == "POST":
        fullname = request.form.get("fullname")
        specialization = request.form.get("specialization")
        hospital = request.form.get("hospital")

        cursor.execute("""
            UPDATE users
            SET fullname=?, specialization=?, hospital=?
            WHERE email=?
        """, (fullname, specialization, hospital, session["email"]))

        conn.commit()
        flash("Profile updated successfully!")

    cursor.execute("""
        SELECT fullname, email, contact, specialization, hospital
        FROM users
        WHERE email=?
    """, (session["email"],))

    user = cursor.fetchone()
    conn.close()

    return render_template("profile.html", user=user)


@app.route("/logout")
def logout():
    if "user" in session:
        add_log(session["user"], "Doctor", "Logout", "Doctor logged out")
    session.clear()
    return redirect(url_for("home"))


# ------------------ ADMIN LOGIN ------------------

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":
            session['admin_logged_in'] = True
            print("ADMIN LOGIN LOG TRIGGERED")
            add_log("Admin", "Admin", "Login", "Admin logged in")
            return redirect('/admin-dashboard')
        else:
            add_log(username, "Admin", "Failed Login", "Invalid admin login attempt")
            return render_template('admin_login.html', error="Invalid Credentials")

    return render_template('admin_login.html')


# ------------------ ADMIN DASHBOARD ------------------

@app.route('/admin-dashboard')
def admin_dashboard():
    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    return render_template('admin_dashboard.html')

@app.route("/admin-users")
def admin_users():

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    search = request.args.get("search", "")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT u.id,
               u.fullname,
               u.email,
               u.hospital,
               COUNT(r.id) as total_uploads,
               u.is_active
        FROM users u
        LEFT JOIN reports r
        ON u.email = r.doctor_email
    """

    params = []

    if search:
        query += " WHERE u.fullname LIKE ? OR u.email LIKE ? OR u.hospital LIKE ?"
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    query += " GROUP BY u.id ORDER BY u.id DESC"

    cursor.execute(query, params)
    users = cursor.fetchall()
    conn.close()

    return render_template("admin_users.html", users=users)


@app.route("/admin-delete-user/<int:user_id>")
def admin_delete_user(user_id):

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get user email first
    cursor.execute("SELECT email FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()

    if user:
        email = user[0]

        # Delete reports of that user
        cursor.execute("DELETE FROM reports WHERE doctor_email=?", (email,))

        # Delete user
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))

        conn.commit()

        add_log("Admin", "Admin", "Delete User",
        f"Deleted user ID {user_id}")

    conn.close()

    return redirect('/admin-users')


@app.route("/admin-toggle-user/<int:user_id>")
def admin_toggle_user(user_id):

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT is_active FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()

    if user:
        new_status = 0 if user[0] == 1 else 1
        cursor.execute("UPDATE users SET is_active=? WHERE id=?", (new_status, user_id))
        conn.commit()

        add_log("Admin", "Admin", "Toggle User Status",
                f"Changed status for user ID {user_id}")    
    
    conn.close()
    return redirect('/admin-users')


# ------------------ ADMIN LOGOUT ------------------

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect('/admin-login')


@app.route("/admin-chart-data")
def admin_chart_data():

    if 'admin_logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---------------- PIE CHART ----------------
    cursor.execute("""
        SELECT prediction, COUNT(*)
        FROM reports
        GROUP BY prediction
    """)
    distribution_data = cursor.fetchall()

    distribution = {
        "NORMAL": 0,
        "VIRAL": 0,
        "BACTERIAL": 0
    }

    for row in distribution_data:
        distribution[row[0]] = row[1]

    # ---------------- WEEKLY UPLOADS ----------------
    cursor.execute("""
        SELECT strftime('%w', date), COUNT(*)
        FROM reports
        WHERE date >= datetime('now','-7 days')
        GROUP BY strftime('%w', date)
    """)
    weekly_data = cursor.fetchall()

    weekly = {str(i): 0 for i in range(7)}

    for row in weekly_data:
        weekly[row[0]] = row[1]

    # ---------------- CONFIDENCE TREND ----------------
    cursor.execute("""
        SELECT strftime('%w', date), AVG(confidence)
        FROM reports
        WHERE date >= datetime('now','-7 days')
        GROUP BY strftime('%w', date)
    """)
    confidence_data = cursor.fetchall()

    confidence = {str(i): 0 for i in range(7)}

    for row in confidence_data:
        confidence[row[0]] = round(row[1], 2)

    conn.close()

    return jsonify({
        "distribution": distribution,
        "weekly": weekly,
        "confidence": confidence
    })

@app.route("/admin-stats")
def admin_stats():

    if 'admin_logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total Users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM reports")
    total_predictions = cursor.fetchone()[0]

    # Normal
    cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='NORMAL'")
    normal = cursor.fetchone()[0]

    # Viral
    cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='VIRAL'")
    viral = cursor.fetchone()[0]

    # Bacterial
    cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='BACTERIAL'")
    bacterial = cursor.fetchone()[0]

    # Today's uploads
    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE date(date)=date('now')
    """)
    today = cursor.fetchone()[0]

    # Low Confidence (< 60%)
    cursor.execute("""
        SELECT COUNT(*) FROM reports 
        WHERE confidence < 60
    """)
    low_conf = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "totalUsers": total_users,
        "totalPredictions": total_predictions,
        "normal": normal,
        "viral": viral,
        "bacterial": bacterial,
        "today": today,
        "lowConfidence": low_conf,
        "model": "CNN-v1"
    })




@app.route("/admin-reports")
def admin_reports():

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.id,
               u.fullname,
               u.email,
               COUNT(r.id) as total_reports
        FROM users u
        LEFT JOIN reports r
        ON u.email = r.doctor_email
        GROUP BY u.id
        ORDER BY total_reports DESC
    """)

    users = cursor.fetchall()
    conn.close()

    return render_template("admin_reports.html", users=users)

@app.route("/admin-user-reports/<int:user_id>")
def admin_user_reports(user_id):

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch doctor info
    cursor.execute("SELECT fullname, email FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return redirect('/admin-reports')

    doctor_name = user[0]
    doctor_email = user[1]

    # STRICT FILTER
    cursor.execute("""
        SELECT id,
               patient_name,
               age,
               gender,
               prediction,
               confidence,
               date
        FROM reports
        WHERE doctor_email = ?
        ORDER BY date DESC
    """, (doctor_email,))

    reports = cursor.fetchall()
    conn.close()

    return render_template(
        "admin_user_reports.html",
        reports=reports,
        doctor_name=doctor_name
    )

@app.route("/admin-delete-report/<int:report_id>")
def admin_delete_report(report_id):

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM reports WHERE id=?", (report_id,))
    conn.commit()
    conn.close()

    add_log("Admin", "Admin", "Delete Report",
        f"Deleted report ID {report_id}")
    
    return redirect('/admin-reports')

@app.route("/admin-logs")
def admin_logs():

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_name,
            role,
            action,
            details,
            datetime(timestamp, '+5 hours', '+30 minutes') as timestamp
        FROM system_logs
        ORDER BY timestamp DESC
    """)

    logs = cursor.fetchall()
    conn.close()

    return render_template("admin_logs.html", logs=logs)

@app.route("/admin-activity-data")
def admin_activity_data():

    if 'admin_logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total logins today
    cursor.execute("""
        SELECT COUNT(*) FROM system_logs
        WHERE action='Login'
        AND date(timestamp)=date('now')
    """)
    total_logins = cursor.fetchone()[0]

    # Failed logins today
    cursor.execute("""
        SELECT COUNT(*) FROM system_logs
        WHERE action='Failed Login'
        AND date(timestamp)=date('now')
    """)
    failed_logins = cursor.fetchone()[0]

    # Predictions today
    cursor.execute("""
        SELECT COUNT(*) FROM reports
        WHERE date(date)=date('now')
    """)
    predictions_today = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "logins": total_logins,
        "failed": failed_logins,
        "predictions": predictions_today
    })

@app.route("/admin-export-logs")
def admin_export_logs():

    if 'admin_logged_in' not in session:
        return redirect('/admin-login')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_name,
               role,
               action,
               details,
               datetime(timestamp, '+5 hours', '+30 minutes')
        FROM system_logs
        ORDER BY timestamp DESC
    """)

    logs = cursor.fetchall()
    conn.close()

    def generate():
        yield "User,Role,Action,Details,Date & Time\n"
        for log in logs:
            row = f"{log[0]},{log[1]},{log[2]},{log[3]},{log[4]}\n"
            yield row

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=system_logs.csv"}
    )

if __name__ == "__main__":
    app.run(debug=True)
