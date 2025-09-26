import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#folder upload
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db = SQLAlchemy(app)

# --------------------- MODELS ---------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    profession = db.Column(db.String(100), nullable=True)
    str = db.Column(db.String(100), nullable=True)
    practice_place = db.Column(db.String(100), nullable=True)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    patient_id = db.Column(db.String(50))
    patient_name = db.Column(db.String(100))
    image_path = db.Column(db.String(200))
    prediction = db.Column(db.String(100))
    accuracy = db.Column(db.Float)
    recommendation = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('histories', lazy=True))

# --------------------- MODEL LOADING ---------------------
model = load_model('model/model-mobilenetv2.h5')
label_map = {
    0: 'Chickenpox',
    1: 'Measles',
    2: 'Monkeypox',
    3: 'Normal'
}

rekomendasi_map = {
    'Monkeypox': '''
    Pasien harus menghindari berbagi handuk mandi atau pakaian dan menggaruk. Lesi
    yang muncul di wajah dan daerah ekstremitas penting dijaga supaya
    tidak digaruk agar tidak menjadi luka. Apabila terdapat papula atau
    pustula yang erosif atau luka lecet bisa diberikan obat topikal antibiotik,
    seperti natrium fusidat salep, asam fusidat krim, mupirocin salep atau
    krim, bila tidak tersedia dapat menggunakan gentamisin salep atau krim.
    Sumber: Kementerian Kesehatan Republik Indonesia. 2023. Pedoman Pencegahan 
    dan Pengendalian Mpox''',

    'Chickenpox': '''
    Istirahat yg cukup dan makan makanan yg bergizi seimbang agar kekebalan tubuh selalu terjaga. 
    Jangan jadikan alasan cacar air untuk tidak mandi.Sementara untuk mengurangi rasa gatal, anda bisa mengkonsumsi 
    obat antihistamin dengan resep dokter atau mengoleskan kalamin diseluruh bintil cacar. 
    Selalu menjaga kebersihan diri saat terkena cacar, ini akan mengurangi resiko terkena infeksi 
    sekunder. Jangan lupa juga tetap perhatikan makanan dan minuman yg dimakan. Jika cacar air ingin cepat pulih, 
    sebaiknya konsumsi asupan yg bergizi tinggi sehingga nutrisi dapat diserap tubuh dan tentunya 
    akan melawan virus-virus tersebut.
    Sumber: Dinas Kesehatan Prov Bangka Belitung. 2023. Penanganan dan Perawatan Cacar Air.''',

    'Measles': '''
    Penyakit ini dapat sembuh dalam beberapa hari tanpa pengobatan. Namun, untuk membantu meredakan gejala, 
    penderita disarankan untuk melakukan upaya-upaya seperti banyak minum air putih, minum obat pereda demam, beristirahat yang cukup,
    mengonsumsi suplemen vitamin A sesuai saran dokter. Campak dapat dicegah dengan pemberian vaksin campak dan dilanjutkan dengan vaksin MMR, yaitu vaksin
    gabungan untuk campak, gondongan, dan rubella.
    Sumber: Kemenkes RS Mohammad Hoesin. Waspadai Penyakit Campak''',
    
    'Normal': 'Tidak ada indikasi penyakit, tetap jaga kesehatan kulit dan kebersihan tubuh.'
}

# --------------------- ROUTES ---------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

# ---------- REGISTER ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Password tidak cocok!')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar!')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        # flash('Registrasi berhasil, silakan login.')
        return redirect(url_for('login'))

    return render_template('register.html')

# ---------- LOGIN ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('detect'))
        else:
            flash('Email atau password salah!')
            return redirect(url_for('login'))

    return render_template('login.html')

# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        #update data diri
        if 'name' in request.form and 'email' in request.form and 'profession' in request.form and 'str' in request.form and 'practice_place' in request.form:
            user.name = request.form['name']
            user.email = request.form['email']
            user.profession = request.form['profession']
            user.str = request.form['str']
            user.practice_place = request.form['practice_place']
            db.session.commit()
            flash('Data diri berhasil diubah!', 'success')
        
        #update password
        elif 'current_password' in request.form and 'new_password' in request.form and 'confirm_password' in request.form:
            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']

            if not check_password_hash(user.password, current_password):
                flash('Password lama salah!', 'danger')
                return redirect(url_for('profile'))
            
            if new_password != confirm_password:
                flash('Password baru dan konfirmasi tidak cocok!', 'danger')
                return redirect(url_for('profile'))

            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash('Password berhasil diubah!', 'success')

        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

# ---------- DETECTION ----------
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        patient_name = request.form['patient_name']
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred_probs = model.predict(img_array)[0]
            pred_index = np.argmax(pred_probs)
            prediction = label_map[pred_index]
            accuracy = round(float(np.max(pred_probs)) * 100, 2)

            recommendation = rekomendasi_map.get(prediction, 'Periksa lebih lanjut.')

            #save to db
            new_history = History(
                patient_id=patient_id,
                patient_name=patient_name,
                image_path=filename,
                prediction=prediction,
                accuracy=accuracy,
                recommendation=recommendation,
                user_id=session['user_id']
            )
            db.session.add(new_history)
            db.session.commit()

            result = {
                'prediction': prediction,
                'accuracy': accuracy,
                'recommendation': recommendation,
                'image_path': filename
            }

    return render_template('detect.html', result=result)

# ---------- HISTORY ----------
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 5

    histories = History.query.filter_by(user_id=session['user_id']).order_by(History.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)

    return render_template('history.html', histories=histories)
    

# --------------------- MAIN ---------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
