from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model_path = 'movie_rating_model.pkl'
model = joblib.load(model_path)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data and convert to float
        year = float(request.form['Year'])
        duration = float(request.form['Duration'])
        votes = float(request.form['Votes'])
        genre_mean_rating = float(request.form['Genre_mean_rating'])
        director_mean_rating = float(request.form['Director_mean_rating'])
        actor1_mean_rating = float(request.form['Actor1_mean_rating'])
        actor2_mean_rating = float(request.form['Actor2_mean_rating'])
        actor3_mean_rating = float(request.form['Actor3_mean_rating'])

        # Create input array as a 2D numpy array
        input_features = np.array([[year, duration, votes, genre_mean_rating,
                                   director_mean_rating, actor1_mean_rating,
                                   actor2_mean_rating, actor3_mean_rating]])

        # Make prediction
        prediction = model.predict(input_features)
        prediction = round(prediction[0], 2)  # Round the prediction for display
    except Exception as e:
        return render_template('index.html', error=str(e))

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
