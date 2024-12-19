from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.cloud import storage

# Authenticate Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Download file from Google Drive
file_id = '1PTHAX9hZX7HBXXUGVvI1ar1LUf4aVbq9'
downloaded_file = drive.CreateFile({'id': file_id})
downloaded_file.GetContentFile('LUMIR_L2R24_TrainVal.zip')

# Upload file to Google Cloud Storage
client = storage.Client()
bucket = client.bucket('prisms_storage')
blob = bucket.blob('LUMIR_L2R24_TrainVal.zip')
blob.upload_from_filename('LUMIR_L2R24_TrainVal.zip')

print("File uploaded to Google Cloud Storage!")
