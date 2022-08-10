from algorithm import *
from flask import Flask, render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    #return "Hello, World!"
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload test")
def salvador():
    return "Hello, Salvador"

@app.route('/json-example', methods=['POST'])
def json_example():
    req_data = request.get_json(force=True)
    language = req_data['language']
    return 'The language value is: {}'.format(language)
    return 'JSON Object Example'

@app.route('/process_data', methods=['POST'])
def process_data():

    #name represente le text saisi, donc il faut le passer en parametre de la fonction de prediction.
    name = request.form['text']

    print(name)

    
    return render_template("index.html", data=name)


@app.route('/embimage', methods=['GET', 'POST'])
def upload_file():
    # check if the post request has the file part
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('Selected files')
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(app.config['UPLOAD_FOLDER'])
            print(filename)

            jpgfile = Image.open(filename)

            # Appeler l'algorithme ici et lui passer en parametres l'image

            print(jpgfile.bits, jpgfile.size, jpgfile.format)
            #return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/text")
def text():
    return "text"


if __name__ == "__main__":
    app.run(debug=True)