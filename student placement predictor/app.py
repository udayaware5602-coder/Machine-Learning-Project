from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

HISTORY_FILE = "prediction_history.csv"


# Create history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(
        columns=[
            "CGPA",
            "IQ",
            "Profile Score",
            "Prediction"
        ]
    )
    df.to_csv(HISTORY_FILE, index=False)


def get_recent_history():
    """Helper function to read recent history safely."""
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        return history.tail(5).values.tolist()

    return []


@app.route("/")
def home():
    recent_history = get_recent_history()

    return render_template(
        "index.html",
        history=recent_history
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        cgpa = float(request.form["cgpa"])
        iq = int(request.form["iq"])
        profile_score = int(request.form["profile_score"])

        # Validation
        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA must be between 0 and 10")

        if iq <= 0:
            raise ValueError("IQ must be greater than 0")

        if profile_score < 0 or profile_score > 100:
            raise ValueError("Profile Score must be between 0 and 100")

        # Prepare input for model
        input_data = np.array(
            [cgpa, iq, profile_score]
        ).reshape(1, 3)

        # Make prediction
        prediction = model.predict(input_data)

        # Assuming:
        # 0 = Placed
        # 1 = Not Placed
        result = "Placed" if prediction[0] == 0 else "Not Placed"

        # Save prediction history
        new_record = pd.DataFrame(
            [[cgpa, iq, profile_score, result]],
            columns=[
                "CGPA",
                "IQ",
                "Profile Score",
                "Prediction"
            ]
        )

        new_record.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

        # Get recent history
        recent_history = get_recent_history()

        return render_template(
            "index.html",
            result=result,
            history=recent_history,
            last_input={
                "cgpa": cgpa,
                "iq": iq,
                "profile_score": profile_score
            }
        )

    except Exception as e:
        recent_history = get_recent_history()

        return render_template(
            "index.html",
            error=str(e),
            history=recent_history
        )


# Run Flask application
if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )