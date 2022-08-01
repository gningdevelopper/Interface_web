# from flask import Flask

# app = Flask(__name__)

# @app.route('/hello')
# def helloIndex():
#     return 'Hello World from Python Flask!'

# app.run(host='0.0.0.0', port=5000)

# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, World!"
    
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    #return "Hello, World!"
    return render_template("index.html")

@app.route("/salvador")
def salvador():
    return "Hello, Salvador"
    

@app.route("/text")
def text():
    return "text"


if __name__ == "__main__":
    app.run(debug=True)