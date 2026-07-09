from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            timestamp=request.form.get("timestamp"),
            temperature=float(request.form.get("temperature")),
            humidity=float(request.form.get("humidity")),
            demand_lag_24hr=float(request.form.get("demand_lag_24hr")),
            demand_lag_168hr=float(request.form.get("demand_lag_168hr")),
            demand_rolling_mean_24hr=float(request.form.get("demand_rolling_mean_24hr")),
            demand_rolling_std_24hr=float(request.form.get("demand_rolling_std_24hr")),
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=round(float(results[0]), 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
