from flask import Flask, render_template, request
import marks as m

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def marks():
    global mk
    mk = 0
    if request.method == "POST":
        sat_score = request.form["sat_score"]
        marks_pred = m.gpa_prediction(sat_score)
        mk = marks_pred
        
    return render_template("index.html", my_marks=mk)


if __name__ == "__main__":
    app.run(debug=True)
