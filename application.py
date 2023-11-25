import sys
from flask import Flask, request, render_template, url_for

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def training():
    train = TrainPipeline()
    tuned_score = train.run_pipeline()
    accuracy = str(round(tuned_score, 2) * 100)[:2]
    message = f"Model trained successfully with accuracy:{accuracy}%"
    return render_template("index.html", message=message)


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            logging.info("Extracting input data...")
            age = request.form.get("age")
            sex = request.form.get("sex")
            bmi = request.form.get("bmi")
            children = request.form.get("children")
            smoker = request.form.get("smoker")
            region = request.form.get("region")

            input_data = CustomData(age, sex, bmi, children, smoker, region)
            dataframe = input_data.get_data_as_dataframe()
            preds = PredictPipeline()
            result = preds.prediction(dataframe)
            return render_template(
                "home.html",
                results="Predicted Insurance Amount ₹ {:.2f}".format(float(result[0])),
            )
        except Exception as e:
            raise CustomException(e, sys)
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
