from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from io import BytesIO
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import numpy as np
from PIL import Image
import shutil
import logging
import zipfile

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


class MedicalImageServer:
    """Complete medical image server with pathology support"""
    
    def __init__(self):
        self.studies_path = 'studies'
    
    def get_dicom_metadata(self, file_path):
        """Extract DICOM metadata"""
        try:
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            return {
                'StudyInstanceUID': str(ds.get('StudyInstanceUID', 'unknown')),
                'PatientName': str(ds.get('PatientName', 'Unknown')),
                'PatientID': str(ds.get('PatientID', 'Unknown')),
                'StudyDate': str(ds.get('StudyDate', '')),
                'Modality': str(ds.get('Modality', 'OT')),
                'PixelSpacing': float(ds.get('PixelSpacing', [0.5])[0]) if ds.get('PixelSpacing') else 0.5
            }
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return None
    
    def get_pixel_array(self, file_path):
        """Extract pixel array from DICOM"""
        try:
            ds = pydicom.dcmread(file_path, force=True)
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                if len(pixel_array.shape) == 3:
                    pixel_array = pixel_array[0]
                return pixel_array
            return None
        except Exception as e:
            logger.error(f"Pixel array extraction failed: {e}")
            return None
    
    def save_dicom(self, file_path, study_uid):
        """Save DICOM file"""
        try:
            study_dir = os.path.join(self.studies_path, study_uid)
            os.makedirs(study_dir, exist_ok=True)
            
            metadata = self.get_dicom_metadata(file_path)
            if metadata:
                metadata['study_uid'] = study_uid
                metadata['created_at'] = datetime.now().isoformat()
                metadata['is_3d'] = False
                
                with open(os.path.join(study_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
            
            dest_path = os.path.join(study_dir, 'image.dcm')
            shutil.copy2(file_path, dest_path)
            
            pixel_array = self.get_pixel_array(file_path)
            if pixel_array is not None:
                if pixel_array.max() > 0:
                    pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
                Image.fromarray(pixel_array).save(os.path.join(study_dir, 'preview.png'))
            
            return True
        except Exception as e:
            logger.error(f"DICOM save failed: {str(e)}")
            return False
    
    def save_image(self, file_path, study_uid, patient_name="Unknown"):
        """Save regular image (PNG, JPG, etc.)"""
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
            
            img.save(os.path.join(study_dir, 'preview.png'))
            
            metadata = {
                'study_uid': study_uid,
                'patient_name': patient_name,
                'modality': 'OT',
                'is_3d': False,
                'pixel_spacing': 0.5,
                'size': list(img.size),
                'created_at': datetime.now().isoformat()
            }
            
            with open(os.path.join(study_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Image save failed: {str(e)}")
            return False
    
    def save_nifti(self, file_path, study_uid, patient_name="Unknown"):
        """Save NIfTI file for 3D visualization"""
        try:
            import nibabel as nib
            
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            
            if len(data.shape) == 4:
                data = data[:, :, :, 0]
            
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data_normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data_normalized = np.zeros_like(data, dtype=np.uint8)
            
            study_dir = os.path.join(self.studies_path, study_uid)
            os.makedirs(study_dir, exist_ok=True)
            
            slices_dir = os.path.join(study_dir, 'slices')
            os.makedirs(slices_dir, exist_ok=True)
            
            num_slices = data_normalized.shape[2]
            for i in range(min(num_slices, 500)):
                slice_img = Image.fromarray(data_normalized[:, :, i])
                slice_img.save(os.path.join(slices_dir, f'slice_{i:04d}.png'))
            
            middle = num_slices // 2
            Image.fromarray(data_normalized[:, :, middle]).save(os.path.join(study_dir, 'preview.png'))
            
            mip = np.max(data_normalized, axis=2)
            Image.fromarray(mip).save(os.path.join(study_dir, 'mip.png'))
            
            metadata = {
                'study_uid': study_uid,
                'patient_name': patient_name,
                'modality': 'MR',
                'is_3d': True,
                'num_slices': num_slices,
                'shape': list(data_normalized.shape),
                'pixel_spacing': float(abs(nii_img.affine[0, 0])) if nii_img.affine.shape[0] > 0 else 1.0,
                'created_at': datetime.now().isoformat()
            }
            
            with open(os.path.join(study_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"NIfTI save failed: {str(e)}")
            return False
    
    def get_all_studies(self):
        """Get list of all studies"""
        studies = []
        if not os.path.exists(self.studies_path):
            return studies
        
        for study_name in os.listdir(self.studies_path):
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
                            'NumberOfInstances': metadata.get('num_slices', 1)
                        })
                    except:
                        studies.append({
                            'StudyInstanceUID': study_name,
                            'PatientName': 'Unknown',
                            'StudyDate': '',
                            'Modality': 'OT',
                            'NumberOfInstances': 1
                        })
                else:
                    studies.append({
                        'StudyInstanceUID': study_name,
                        'PatientName': 'Unknown',
                        'StudyDate': '',
                        'Modality': 'OT',
                        'NumberOfInstances': 1
                    })
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
                        'modality': metadata.get('modality', 'OT')
                    }
            except:
                pass
        return {'pixel_spacing': 0.5, 'is_3d': False, 'slices': 1, 'patient_name': 'Unknown', 'modality': 'OT'}
    
    def get_preview_path(self, study_uid):
        """Get preview image path"""
        study_path = os.path.join(self.studies_path, study_uid)
        preview_path = os.path.join(study_path, 'preview.png')
        if os.path.exists(preview_path):
            return preview_path
        return None
    
    def get_first_dicom_path(self, study_uid):
        """Get first DICOM file path"""
        study_path = os.path.join(self.studies_path, study_uid)
        if os.path.exists(study_path):
            for file in os.listdir(study_path):
                if file.endswith('.dcm'):
                    return os.path.join(study_path, file)
        return None
    
    def get_original_image_path(self, study_uid):
        """Get original image path"""
        study_path = os.path.join(self.studies_path, study_uid)
        for file in os.listdir(study_path):
            if file.endswith(('.dcm', '.png', '.jpg', '.jpeg')):
                return os.path.join(study_path, file)
        return None
    
    def get_mip_path(self, study_uid):
        """Get MIP path for 3D volume"""
        study_path = os.path.join(self.studies_path, study_uid)
        mip_path = os.path.join(study_path, 'mip.png')
        if os.path.exists(mip_path):
            return mip_path
        return None
    
    def get_slice_path(self, study_uid, slice_idx):
        """Get specific slice path for 3D volume"""
        study_path = os.path.join(self.studies_path, study_uid)
        slices_dir = os.path.join(study_path, 'slices')
        if os.path.exists(slices_dir):
            slice_path = os.path.join(slices_dir, f'slice_{slice_idx:04d}.png')
            if os.path.exists(slice_path):
                return slice_path
        return None
    
    def delete_study(self, study_uid):
        """Delete a study"""
        study_path = os.path.join(self.studies_path, study_uid)
        if os.path.exists(study_path):
            shutil.rmtree(study_path)
            return True
        return False

server = MedicalImageServer()


# ==================== EXPORT FUNCTIONS ====================

def create_dicom_sr(study_uid, annotations, measurements, cell_count):
    """Create DICOM Structured Report from annotations"""
    try:
        # Get study info
        study_info = server.get_study_info(study_uid)
        
        # Create SR dataset
        sr_ds = Dataset()
        sr_ds.file_meta = Dataset()
        sr_ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # Patient information
        sr_ds.PatientName = study_info.get('patient_name', 'Unknown')
        sr_ds.PatientID = study_uid[:20]
        
        # Study information
        sr_ds.StudyInstanceUID = generate_uid()
        sr_ds.StudyDate = datetime.now().strftime('%Y%m%d')
        sr_ds.StudyTime = datetime.now().strftime('%H%M%S')
        sr_ds.StudyDescription = f"Pathology Report for {study_uid}"
        
        # Series information
        sr_ds.SeriesInstanceUID = generate_uid()
        sr_ds.Modality = 'SR'
        sr_ds.SeriesDescription = "Pathology Measurements and Annotations"
        
        # SOP information
        sr_ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR
        sr_ds.SOPInstanceUID = generate_uid()
        
        # Create content sequence
        content_seq = []
        
        # Add measurement data
        for measurement in measurements:
            item = Dataset()
            item.ValueType = "NUM"
            
            # Concept name
            concept = Dataset()
            concept.CodeValue = "121207"
            concept.CodingSchemeDesignator = "DCM"
            concept.CodeMeaning = "Measurement"
            item.ConceptNameCodeSequence = [concept]
            
            # Measured value
            measured = Dataset()
            measured.NumericValue = float(measurement.get('value', 0))
            
            # Units
            units = Dataset()
            if measurement.get('unit') == 'mm':
                units.CodeValue = "mm"
                units.CodeMeaning = "Millimeter"
            elif measurement.get('unit') == 'degrees':
                units.CodeValue = "deg"
                units.CodeMeaning = "Degree"
            elif measurement.get('unit') == 'mm2':
                units.CodeValue = "mm2"
                units.CodeMeaning = "Square Millimeter"
            else:
                units.CodeValue = "1"
                units.CodeMeaning = "count"
            units.CodingSchemeDesignator = "UCUM"
            
            measured.MeasurementUnitsCodeSequence = [units]
            item.MeasuredValueSequence = [measured]
            
            content_seq.append(item)
        
        # Add cell count
        if cell_count > 0:
            cell_item = Dataset()
            cell_item.ValueType = "NUM"
            
            concept = Dataset()
            concept.CodeValue = "122367"
            concept.CodingSchemeDesignator = "DCM"
            concept.CodeMeaning = "Cell Count"
            cell_item.ConceptNameCodeSequence = [concept]
            
            measured = Dataset()
            measured.NumericValue = float(cell_count)
            
            units = Dataset()
            units.CodeValue = "1"
            units.CodeMeaning = "count"
            units.CodingSchemeDesignator = "UCUM"
            measured.MeasurementUnitsCodeSequence = [units]
            
            cell_item.MeasuredValueSequence = [measured]
            content_seq.append(cell_item)
        
        # Add annotation descriptions
        for ann in annotations:
            if ann.get('type') not in ['label', 'head', 'arrowhead']:
                ann_item = Dataset()
                ann_item.ValueType = "TEXT"
                
                concept = Dataset()
                concept.CodeValue = "111060"
                concept.CodingSchemeDesignator = "DCM"
                concept.CodeMeaning = "Annotation"
                ann_item.ConceptNameCodeSequence = [concept]
                
                ann_item.TextValue = f"{ann.get('type', 'unknown')} annotation"
                content_seq.append(ann_item)
        
        sr_ds.ContentSequence = content_seq
        
        # Save SR file
        export_dir = os.path.join('exports', study_uid)
        os.makedirs(export_dir, exist_ok=True)
        sr_path = os.path.join(export_dir, f'sr_{study_uid}.dcm')
        sr_ds.save_as(sr_path, write_like_original=False)
        
        return sr_path
    except Exception as e:
        logger.error(f"SR creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_json_export(study_uid, annotations, measurements, cell_count, study_info):
    """Create JSON export of all data"""
    export_data = {
        'export_date': datetime.now().isoformat(),
        'study_uid': study_uid,
        'study_info': study_info,
        'cell_count': cell_count,
        'measurements': measurements,
        'annotations': [
            {
                'id': ann.get('id'),
                'type': ann.get('type'),
                'color': ann.get('color'),
                'timestamp': datetime.now().isoformat()
            }
            for ann in annotations if ann.get('type') not in ['label', 'head', 'arrowhead']
        ],
        'export_format_version': '1.0'
    }
    
    export_dir = os.path.join('exports', study_uid)
    os.makedirs(export_dir, exist_ok=True)
    json_path = os.path.join(export_dir, f'export_{study_uid}.json')
    
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return json_path


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
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
            elif ext in ['nii', 'gz']:
                if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                    success = server.save_nifti(temp_path, study_uid, patient_name)
            elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                success = server.save_image(temp_path, study_uid, patient_name)
            
            if success:
                uploaded_studies.append(study_uid)
                logger.info(f"Uploaded: {filename} -> {study_uid}")
            
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
    """Get all studies"""
    try:
        studies = server.get_all_studies()
        return jsonify({'success': True, 'studies': studies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/image', methods=['GET'])
def get_study_image(study_uid):
    """Get study preview image"""
    try:
        preview_path = server.get_preview_path(study_uid)
        
        if not preview_path:
            dicom_path = server.get_first_dicom_path(study_uid)
            if dicom_path:
                pixel_array = server.get_pixel_array(dicom_path)
                if pixel_array is not None:
                    if pixel_array.max() > 0:
                        pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
                    img = Image.fromarray(pixel_array)
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    return send_file(img_bytes, mimetype='image/png')
        
        if preview_path:
            return send_file(preview_path, mimetype='image/png')
        
        return jsonify({'error': 'No image found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/info', methods=['GET'])
def get_study_info(study_uid):
    """Get study information"""
    try:
        info = server.get_study_info(study_uid)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/mip', methods=['GET'])
def get_study_mip(study_uid):
    """Get MIP for 3D volume"""
    try:
        mip_path = server.get_mip_path(study_uid)
        if mip_path:
            return send_file(mip_path, mimetype='image/png')
        return jsonify({'error': 'No MIP available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/slice/<int:slice_idx>', methods=['GET'])
def get_study_slice(study_uid, slice_idx):
    """Get specific slice from 3D volume"""
    try:
        slice_path = server.get_slice_path(study_uid, slice_idx)
        if slice_path:
            return send_file(slice_path, mimetype='image/png')
        return jsonify({'error': 'Slice not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>', methods=['DELETE'])
def delete_study(study_uid):
    """Delete a study"""
    try:
        success = server.delete_study(study_uid)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study/<study_uid>/annotations', methods=['GET', 'POST'])
def handle_annotations(study_uid):
    """Store and retrieve annotations"""
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
    """Export study data as DICOM SR and JSON"""
    try:
        data = request.json
        annotations = data.get('annotations', [])
        measurements = data.get('measurements', [])
        cell_count = data.get('cell_count', 0)
        
        # Get study info
        study_info = server.get_study_info(study_uid)
        
        # Create DICOM SR
        sr_path = create_dicom_sr(study_uid, annotations, measurements, cell_count)
        
        # Create JSON export
        json_path = create_json_export(study_uid, annotations, measurements, cell_count, study_info)
        
        # Get original image
        original_image = server.get_original_image_path(study_uid)
        
        # Create zip file with all exports
        zip_path = os.path.join('exports', study_uid, f'{study_uid}_complete_export.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            if sr_path and os.path.exists(sr_path):
                zipf.write(sr_path, os.path.basename(sr_path))
            if json_path and os.path.exists(json_path):
                zipf.write(json_path, os.path.basename(json_path))
            if original_image and os.path.exists(original_image):
                zipf.write(original_image, os.path.basename(original_image))
        
        return send_file(zip_path, as_attachment=True, download_name=f'{study_uid}_export.zip')
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/dicom/<study_uid>', methods=['GET'])
def export_dicom_only(study_uid):
    """Export only DICOM SR"""
    try:
        # Get annotations from file
        annotations_file = os.path.join('studies', study_uid, 'annotations.json')
        annotations = []
        measurements = []
        cell_count = 0
        
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
                measurements = data.get('measurements', [])
                cell_count = data.get('cell_count', 0)
        
        sr_path = create_dicom_sr(study_uid, annotations, measurements, cell_count)
        
        if sr_path and os.path.exists(sr_path):
            return send_file(sr_path, as_attachment=True, download_name=f'{study_uid}_sr.dcm')
        
        return jsonify({'error': 'Failed to create DICOM SR'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/json/<study_uid>', methods=['GET'])
def export_json_only(study_uid):
    """Export only JSON"""
    try:
        annotations_file = os.path.join('studies', study_uid, 'annotations.json')
        annotations = []
        measurements = []
        cell_count = 0
        
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
                measurements = data.get('measurements', [])
                cell_count = data.get('cell_count', 0)
        
        study_info = server.get_study_info(study_uid)
        json_path = create_json_export(study_uid, annotations, measurements, cell_count, study_info)
        
        if json_path and os.path.exists(json_path):
            return send_file(json_path, as_attachment=True, download_name=f'{study_uid}_export.json')
        
        return jsonify({'error': 'Failed to create JSON export'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get viewer configuration"""
    return jsonify({
        'version': '3.0',
        'features': ['annotations', 'measurements', 'segmentation', 'pathology', '3d', 'dicom_export', 'json_export']
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 PATHOLOGY MEDICAL IMAGING VIEWER")
    print("="*60)
    print(f"📱 Access: http://127.0.0.1:5000")
    print(f"📁 Studies stored in: studies/")
    print(f"📤 Exports stored in: exports/")
    print("\n✅ EXPORT FEATURES:")
    print("   • DICOM Structured Report (SR)")
    print("   • JSON Export")
    print("   • Complete ZIP Export")
    print("   • Original Image Export")
    print("\n✅ MEASUREMENT FEATURES:")
    print("   • Unlimited Zoom (10% - 1000%)")
    print("   • Measurements in Microns (µm)")
    print("   • Cell Counter with Auto-numbering")
    print("   • Area Measurement (µm²)")
    print("   • Angle & Bi-directional Measurements")
    print("   • Nuclei Detection Segmentation")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)