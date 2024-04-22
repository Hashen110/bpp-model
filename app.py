from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('model.h5')  # Path to trained model

# Load the MinMaxScaler used in training
scaler = MinMaxScaler()
scaler.fit([[0], [1]])  # Adjust to the feature range and shape you used in training

df = pd.read_csv('output.csv')  # Path to dataset
time_step = 15  # Adjust according to the time step used in training


# Define a function to predict future prices
def predict_future_days(pred_days):
    # Get the last known data for predictions (adjust according to data structure)
    last_known_data = df['Close'][-time_step:].values  # Load the last data from dataset

    # Convert the data to a list if it's in a different format
    last_known_data = last_known_data.tolist()

    # Initialize the list to store future predictions
    lst_output = []

    # Predict future prices
    for i in range(pred_days):
        if len(last_known_data) > time_step:
            # Prepare input for the model
            x_input = np.array(last_known_data[-time_step:])
            x_input = x_input.reshape((1, time_step, 1))

            # Make prediction
            yhat = model.predict(x_input, verbose=0)

            # Update the data with the prediction
            last_known_data.append(yhat[0].tolist()[0])

            # Add the prediction to the output list
            lst_output.append(yhat[0].tolist()[0])
        else:
            # If not enough data, reshape the input and make prediction
            x_input = np.array(last_known_data).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)

            # Add the prediction to the data and output list
            last_known_data.append(yhat[0].tolist()[0])
            lst_output.append(yhat[0].tolist()[0])

    # Convert the predictions back to original scale
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    return lst_output


# Create a Flask app
def create_app():
    app = Flask(__name__)

    # Define a route for getting predictions
    @app.route('/predict', methods=['GET'])
    def predict():
        # Get the number of days to predict from the query parameters
        pred_days = int(request.args.get('pred_days', 30))  # Default to 30 days if not provided

        # Make predictions
        predictions = predict_future_days(pred_days)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions})

    # 404 Error handling
    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify(error=404, text=str(e)), 404

    return app


# Run the app
if __name__ == '__main__':
    create_app().run(debug=True)
