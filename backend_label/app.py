from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import json

UPLOAD_FOLDER = '/media/4tbdrive/engines/cs230/backend_label/uploads/'
ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.secret_key = os.urandom(24)  # Set a secret key for session management

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON data received"}), 400

        folder = request.headers.get('Player').split(".")[0]
        if not os.path.exists(UPLOAD_FOLDER + '/' + folder):
            os.makedirs(UPLOAD_FOLDER + '/' + folder)

        filename = request.headers.get('Shot') + '.json'
        file_path = os.path.join(UPLOAD_FOLDER, folder, filename)

        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

        return jsonify({"message": "File successfully uploaded"}), 200


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)