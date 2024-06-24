import joblib
from flask import Flask
from sklearn.linear_model import LinearRegression

# Create Flask app
app = Flask(__name__)

# Load the model
model_path = 'E:\movie_rating/movie_rating_model.pkl'
loaded_model = joblib.load(model_path)

# Check the type of the loaded model
print(f"Model type: {type(loaded_model)}")

# Route for checking the model
@app.route('/check_model')
def check_model():
    return f"Model type: {type(loaded_model)}"

if __name__ == '__main__':
    app.run(debug=True)
