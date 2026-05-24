import os

class Config:
    # Production settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # File settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    
    # Get absolute paths for Render
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Use /data for persistent storage on Render
    if os.environ.get('RENDER'):
        UPLOAD_FOLDER = '/opt/render/project/src/data/uploads'
        STUDIES_FOLDER = '/opt/render/project/src/data/studies'
        DATABASE_FOLDER = '/opt/render/project/src/data/database'
        EXPORTS_FOLDER = '/opt/render/project/src/data/exports'
        CACHE_FOLDER = '/opt/render/project/src/data/cache'
    else:
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
        STUDIES_FOLDER = os.path.join(BASE_DIR, 'studies')
        DATABASE_FOLDER = os.path.join(BASE_DIR, 'database')
        EXPORTS_FOLDER = os.path.join(BASE_DIR, 'exports')
        CACHE_FOLDER = os.path.join(BASE_DIR, 'cache')
    
    # Allowed file types
    ALLOWED_EXTENSIONS = {'dcm', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = False
    
    # Create folders
    for folder in [UPLOAD_FOLDER, STUDIES_FOLDER, DATABASE_FOLDER, EXPORTS_FOLDER, CACHE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
