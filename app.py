from flask import Flask, request
from flask import render_template

# creates the app
app = Flask(__name__)

# Home page
@app.route("/")
def index():
    """
    It returns the rendered template "index.html" with the variable "result" set to None
    
    Returns:
      The index.html file is being returned.
    """
    result = None
    return render_template("index.html", result=result)

# prediction page
@app.route("/estimate", methods=["POST"])
def estimate():
    result = 'ok'
    return render_template("index.html", result=result)

if __name__=="__main__":
    app.run(host='localhost', port=3000, debug=True)

    