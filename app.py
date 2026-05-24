from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from io import BytesIO
import pydicom
import numpy as np
from PIL import Image
import shutil
import logging
import zipfile
import threading
import hashlib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Create necessary folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('studies', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('exports', exist_ok=True)
os.makedirs('cache', exist_ok=True)


class MedicalImageServer:
    """Complete medical image server"""
    
    def __init__(self):
        self.studies_path = 'studies'
        self.cache_path = 'cache'
        self.preview_cache = {}
        
    def get_dicom_metadata_fast(self, file_path):
        """Fast metadata extraction"""
        try:
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            return {
                'StudyInstanceUID': str(ds.get('StudyInstanceUID', 'unknown')),
                'PatientName': str(ds.get('PatientName', 'Unknown')),
                'PatientID': str(ds.get('PatientID', 'Unknown')),
                'StudyDate': str(ds.get('StudyDate', '')),
                'Modality': str(ds.get('Modality', 'OT')),
                'PixelSpacing': float(ds.get('PixelSpacing', [0.5])[0]) if ds.get('PixelSpacing') else 0.5,
                'Rows': int(ds.get('Rows', 0)),
                'Columns': int(ds.get('Columns', 0))
            }
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return None
    
    def generate_preview_fast(self, file_path, max_size=512):
        """Generate preview quickly"""
        try:
            cache_key = hashlib.md5(f"{file_path}_{max_size}".encode()).hexdigest()
            cache_file = os.path.join(self.cache_path, f"{cache_key}.png")
            
            if os.path.exists(cache_file):
                return cache_file
            
            ds = pydicom.dcmread(file_path, force=True)
            
            if not hasattr(ds, 'pixel_array'):
                return None
            
            pixel_array = ds.pixel_array
            
            if len(pixel_array.shape) == 3:
                pixel_array = pixel_array[0]
            
            h, w = pixel_array.shape
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                step_h = h // new_h
                step_w = w // new_w
                if step_h > 0 and step_w > 0:
                    pixel_array = pixel_array[::step_h, ::step_w]
            
            if pixel_array.max() > 0:
                pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
            
            img = Image.fromarray(pixel_array)
            img.save(cache_file, 'PNG', optimize=True)
            
            return cache_file
            
        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            return None
    
    def save_dicom(self, file_path, study_uid):
        """Save DICOM file"""
        try:
            study_dir = os.path.join(self.studies_path, study_uid)
            os.makedirs(study_dir, exist_ok=True)
            
            metadata = self.get_dicom_metadata_fast(file_path)
            if metadata:
                metadata['study_uid'] = study_uid
                metadata['created_at'] = datetime.now().isoformat()
                metadata['is_3d'] = False
                metadata['processing'] = False
                
                with open(os.path.join(study_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
            
            dest_path = os.path.join(study_dir, 'image.dcm')
            shutil.copy2(file_path, dest_path)
            
            # Generate preview in background
            def generate_preview():
                preview_path = self.generate_preview_fast(dest_path)
                if preview_path:
                    shutil.copy2(preview_path, os.path.join(study_dir, 'preview.png'))
            
            thread = threading.Thread(target=generate_preview)
            thread.start()
            
            return True
        except Exception as e:
            logger.error(f"DICOM save failed: {str(e)}")
            return False
    
    def save_image(self, file_path, study_uid, patient_name="Unknown"):
        """Save regular image"""
        try:
            study_dir = os.path.join(self.studies_path, study_uid)
            os.makedirs(study_dir, exist_ok=True)
            
            img = Image.open(file_path)
            
            if img.mode == 'RGB':
                img = img.convert('L')
            elif img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = rgb_img.convert('L')
            
            preview = img.copy()
            preview.thumbnail((512, 512))
            preview.save(os.path.join(study_dir, 'preview.png'))
            
            metadata = {
                'study_uid': study_uid,
                'patient_name': patient_name,
                'modality': 'OT',
                'is_3d': False,
                'pixel_spacing': 0.5,
                'size': list(img.size),
                'created_at': datetime.now().isoformat(),
                'processing': False
            }
            
            with open(os.path.join(study_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Image save failed: {str(e)}")
            return False
    
    def get_all_studies(self, limit=50):
        """Get all studies"""
        studies = []
        if not os.path.exists(self.studies_path):
            return studies
        
        for study_name in sorted(os.listdir(self.studies_path), reverse=True)[:limit]:
            study_path = os.path.join(self.studies_path, study_name)
            if os.path.isdir(study_path):
                metadata_file = os.path.join(study_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        studies.append({
                            'StudyInstanceUID': study_name,
                            'PatientName': metadata.get('patient_name', metadata.get('PatientName', 'Unknown')),
                            'StudyDate': metadata.get('study_date', metadata.get('StudyDate', '')),
                            'Modality': metadata.get('modality', 'OT'),
                            'NumberOfInstances': 1,
                            'processing': metadata.get('processing', False)
                        })
                    except:
                        continue
        return studies
    
    def get_study_info(self, study_uid):
        """Get study information"""
        study_path = os.path.join(self.studies_path, study_uid)
        metadata_file = os.path.join(study_path, 'metadata.json')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return {
                        'pixel_spacing': metadata.get('pixel_spacing', 0.5),
                        'is_3d': metadata.get('is_3d', False),
                        'slices': metadata.get('num_slices', 1),
                        'patient_name': metadata.get('patient_name', 'Unknown'),
                        'modality': metadata.get('modality', 'OT'),
                        'processing': metadata.get('processing', False)
                    }
            except:
                pass
        return {'pixel_spacing': 0.5, 'is_3d': False, 'slices': 1, 'patient_name': 'Unknown', 'modality': 'OT', 'processing': False}
    
    def get_preview(self, study_uid):
        """Get preview image"""
        study_path = os.path.join(self.studies_path, study_uid)
        preview_path = os.path.join(study_path, 'preview.png')
        
        if os.path.exists(preview_path):
            return preview_path
        
        dicom_path = os.path.join(study_path, 'image.dcm')
        if os.path.exists(dicom_path):
            preview = self.generate_preview_fast(dicom_path)
            if preview:
                shutil.copy2(preview, preview_path)
                return preview_path
        
        return None
    
    def get_original_image_path(self, study_uid):
        """Get original image path"""
        study_path = os.path.join(self.studies_path, study_uid)
        for file in os.listdir(study_path):
            if file.endswith(('.dcm', '.png', '.jpg', '.jpeg')):
                return os.path.join(study_path, file)
        return None
    
    def delete_study(self, study_uid):
        """Delete a study"""
        study_path = os.path.join(self.studies_path, study_uid)
        if os.path.exists(study_path):
            shutil.rmtree(study_path)
            return True
        return False

server = MedicalImageServer()


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        files = request.files.getlist('files')
        patient_name = request.form.get('patient_name', 'Unknown')
        
        uploaded_studies = []
        
        for file in files:
            if not file or not file.filename:
                continue
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join('uploads', filename)
            file.save(temp_path)
            
            study_uid = f"{int(datetime.now().timestamp())}_{len(uploaded_studies)}"
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            success = False
            if ext == 'dcm':
                success = server.save_dicom(temp_path, study_uid)
            elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                success = server.save_image(temp_path, study_uid, patient_name)
            
            if success:
                uploaded_studies.append(study_uid)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'uploaded': uploaded_studies,
            'studies': server.get_all_studies()
        })
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/studies', methods=['GET'])
def get_studies():
    try:
        studies = server.get_all_studies()
        return jsonify({'success': True, 'studies': studies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/image', methods=['GET'])
def get_study_image(study_uid):
    try:
        preview_path = server.get_preview(study_uid)
        if preview_path and os.path.exists(preview_path):
            return send_file(preview_path, mimetype='image/png')
        return jsonify({'error': 'No image found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/info', methods=['GET'])
def get_study_info(study_uid):
    try:
        info = server.get_study_info(study_uid)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>', methods=['DELETE'])
def delete_study(study_uid):
    try:
        success = server.delete_study(study_uid)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/annotations', methods=['GET', 'POST'])
def handle_annotations(study_uid):
    annotations_file = os.path.join('studies', study_uid, 'annotations.json')
    
    if request.method == 'GET':
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'annotations': []})
    
    elif request.method == 'POST':
        annotations = request.json
        annotations['last_modified'] = datetime.now().isoformat()
        os.makedirs(os.path.join('studies', study_uid), exist_ok=True)
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        return jsonify({'success': True})


@app.route('/api/export/<study_uid>', methods=['POST'])
def export_study(study_uid):
    try:
        data = request.json
        annotations = data.get('annotations', [])
        measurements = data.get('measurements', [])
        cell_count = data.get('cell_count', 0)
        
        study_info = server.get_study_info(study_uid)
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'study_uid': study_uid,
            'study_info': study_info,
            'cell_count': cell_count,
            'measurements': measurements,
            'annotations': annotations,
            'export_format_version': '1.0'
        }
        
        export_dir = os.path.join('exports', study_uid)
        os.makedirs(export_dir, exist_ok=True)
        json_path = os.path.join(export_dir, f'export_{study_uid}.json')
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return send_file(json_path, as_attachment=True, download_name=f'{study_uid}_export.json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'version': '3.0',
        'features': ['annotations', 'measurements', 'segmentation', 'pathology', 'json_export']
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 PATHOLOGY MEDICAL IMAGING VIEWER")
    print("="*60)
    print(f"📱 Access: http://127.0.0.1:5000")
    print("\n✅ FEATURES:")
    print("   • User-selectable units (µm, mm, cm, pixels)")
    print("   • Cell counter with auto-numbering")
    print("   • Measurements in any unit")
    print("   • Unlimited Zoom (10% - 1000%)")
    print("   • Angle & Bi-directional measurements")
    print("   • JSON Export")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
