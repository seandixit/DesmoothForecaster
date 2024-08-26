import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import seaborn as sns
import matplotlib.pyplot as plt

class DesmoothForecasterModel:
    """
    A class to create and train a model that uses smoothed time-series data for forecasting.
    The model supports LSTM or custom neural network architectures and can apply exponential 
    smoothing or moving average smoothing to the target variable before training.

    Attributes:
    ----------
    model_type : str
        The type of model to use ('LSTM' or 'custom').
    custom_model : keras.Model, optional
        A custom model to use instead of the default LSTM model.
    smoothing : str
        The smoothing technique to apply ('exp_smoothing' or 'moving_average').
    smoothing_param : float or int
        The parameter for the chosen smoothing method. For exponential smoothing, 
        this is the smoothing level (alpha). For moving average, this is the window size.
    scaler : StandardScaler
        A scaler instance used to normalize the data.
    model : keras.Model, optional
        The neural network model to be trained.
    trainX : np.ndarray, optional
        The input training data prepared for the model.
    df_for_training : pd.DataFrame, optional
        The DataFrame containing smoothed and normalized data used for training.
    n_future : int, optional
        The number of future timesteps the model is trained to predict.
    Y : pd.Series, optional
        The original target variable before smoothing.
    lookback : int, optional
        The number of past timesteps used to predict the future.
    """

    def __init__(self, model_type='LSTM', custom_model=None, smoothing="exp_smoothing", smoothing_param=-1):
        """
        Initializes the DesmoothForecasterModel with the specified parameters.

        Parameters:
        ----------
        model_type : str, optional
            The type of model to use ('LSTM' by default, or 'custom').
        custom_model : keras.Model, optional
            A custom model to use instead of the default LSTM model_type (default is None). model_type must be 'custom'.
        smoothing : str, optional
            The smoothing technique to apply ('exp_smoothing' by default, or 'moving_average').
        smoothing_param : float or int, optional
            The parameter for the smoothing method. Default is -1, which auto-sets the value 
            based on the smoothing method: 0.01 for exponential smoothing and 10 for moving average.

        Raises:
        ------
        ValueError
            If an invalid smoothing method is provided.
        """
        self.model_type = model_type
        self.custom_model = custom_model
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param
        self.scaler = StandardScaler()
        self.model = None
        self.trainX = None
        self.df_for_training = None
        self.n_future = None
        self.Y = None
        self.lookback = None

        # Adjust smoothing_param if not set
        if self.smoothing_param == -1:
            if self.smoothing == 'exp_smoothing':
                self.smoothing_param = 0.01
            elif self.smoothing == 'moving_average':
                self.smoothing_param = 10

    def smooth_data(self, Y):
        """
        Applies the chosen smoothing technique to the target variable.

        Parameters:
        ----------
        Y : pd.Series
            The target variable to be smoothed.

        Returns:
        -------
        pd.Series
            The smoothed target variable.
        """
        if self.smoothing == 'exp_smoothing':
            exp_smooth_model = SimpleExpSmoothing(Y)
            fit = exp_smooth_model.fit(smoothing_level=self.smoothing_param, optimized=False)
            Y = fit.fittedvalues
        elif self.smoothing == 'moving_average':
            Y = Y.rolling(window=self.smoothing_param).mean()

        return Y

    def prepare_data(self, time_column, X, Y, lookback, n_future) -> pd.DataFrame:
        """
        Prepares the input and output data for training the model by applying smoothing, 
        normalizing, and reshaping it into sequences.

        Parameters:
        ----------
        time_column : pd.Series or pd.Index
            The time indices or datetime values for the data.
        X : pd.DataFrame
            The input features (e.g., df[features]).
        Y : pd.Series
            The target variable (e.g., df['PW']).
        lookback : int
            The number of past timesteps used to predict the future (e.g., 10).
        n_future : int
            The number of future timesteps to predict.

        Returns:
        -------
        np.ndarray
            The prepared output data (target variable) for training.

        Raises:
        ------
        ValueError
            If not enough data is available to create training samples.
        """
        # Set data
        self.Y = Y
        # Smooth the data
        Y = self.smooth_data(Y)

        # Create training dataframe
        df_for_training = pd.concat([Y, X], axis=1)
        df_for_training['time_column'] = time_column
        df_for_training = df_for_training.set_index('time_column')
        df_for_training = df_for_training.rename(columns={df_for_training.columns[0]: "smoothed"})

        # Normalize the dataset
        df_for_training_scaled = self.scaler.fit_transform(df_for_training)
        
        # Create training data
        trainX, trainY = [], []
        for i in range(lookback, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - lookback:i, 0:df_for_training.shape[1]])
            trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)
        
        # Ensure data consistency
        assert trainX.shape[0] > 0, "Not enough data to create training samples."
        assert trainX.shape[0] == trainY.shape[0], "Mismatch between trainX and trainY in number of samples."

        # Save instance variables
        self.trainX = trainX
        self.df_for_training = df_for_training

        return trainY

    def build_model(self, input_shape) -> Sequential:
        """
        Builds the neural network model based on the specified model type.

        For an LSTM model, it creates a sequential model with two LSTM layers, 
        followed by dense layers. If a custom model is specified, it will use that instead.

        Returns:
        -------
        keras.Model
            The compiled neural network model.

        Raises:
        ------
        ValueError
            If neither a default LSTM nor a custom model is provided.
        """
        if self.model_type == "LSTM":
            # Default LSTM Model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            print(model.summary())
            self.model = model
        elif self.model_type == "custom":
            self.model = self.custom_model
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        return self.model

    def train_model(self, time_column, X, Y, lookback, n_future, epochs, batch_size=32, validation_split=0.2) -> Sequential:
        """
        Trains the neural network model on the prepared data.

        Parameters:
        ----------
        X : np.ndarray
            The input data for training, prepared by the prepare_data method.
        y : np.ndarray
            The target variable for training.
        model : keras.Model, optional
            A custom model to use instead of the default LSTM model (default is None).
        epochs : int, optional
            The number of epochs to train the model (default is 100).

        Returns:
        -------
        keras.Model
            The trained neural network model.
        """
        trainY = self.prepare_data(time_column, X, Y, lookback, n_future)
        self.build_model((self.trainX.shape[1], self.trainX.shape[2]))
        self.lookback = lookback
        self.n_future = n_future

        # Train model
        history = self.model.fit(self.trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return self.model

    def predict(self, data=None, n_timesteps=10) -> pd.DataFrame:
        """
        Predicts future values based on the trained model and prepared data. 

        Parameters:
        ----------
        data: pd.DataFrame
            The data to use for prediction. If not provided, the last n_timesteps of the training data will be used.
        n_timesteps : int
            Number of time steps to show forecast for. So, if n_future is 5 and n_timesteps is 20, you will have 4 series of predictions displayed.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the actual and predicted values with the corresponding time indices.
        """
        if data is None:
            data = self.df_for_training
        
        if self.trainX is None or data is None:
            raise ValueError("Model must be trained before predicting.")

        forecast_period_dates = data.index[-(n_timesteps):]
        forecast = self.model.predict(self.trainX[-n_timesteps:])

        forecast_copies = np.repeat(forecast, data.shape[1], axis=-1)
        y_pred_future = self.scaler.inverse_transform(forecast_copies)[:, 0]  

        forecast_dates = [time_i for time_i in forecast_period_dates]
        df_forecast = pd.DataFrame({'Time': np.array(forecast_dates), 'Predicted Target': y_pred_future})
        df_forecast['Time'] = pd.to_datetime(df_forecast['Time'])

        original = data['smoothed'].to_frame()
        original['Time'] = pd.to_datetime(data.index)

        # Calculate the time interval between timesteps to shift (n_future-1) timesteps to the left (alignment)
        time_interval = df_forecast['Time'].diff().mean()
        df_forecast['Time'] = df_forecast['Time'] - (self.n_future - 1) * time_interval
        original = original[:-(self.n_future-1)]

        # Slice the original DataFrame to get the last 2.5% of rows
        last_percent_index = int(len(original) * 0.975)
        last_percent_original = original.iloc[last_percent_index:].copy()

        # Plot the results
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=last_percent_original, x='Time', y='smoothed', label='Actual')
        sns.lineplot(data=df_forecast, x='Time', y='Predicted Target', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Smoothed Time-series Prediction')
        plt.legend()
        plt.show()

        # Now desmooth the smoothed predictions
        smoothed_predictions = df_forecast['Predicted Target']
        smoothed_actual = self.df_for_training['smoothed'][self.lookback:self.lookback + len(smoothed_predictions)]
        unsmoothed_actual = self.Y.to_frame()
        unsmoothed_predictions = []

        if (self.smoothing == 'exp_smoothing'):
          for i in range(len(smoothed_predictions)-1):
            unsmoothed_predictions.append((smoothed_predictions[i+1] - (1 - self.smoothing_param) * smoothed_predictions[i])/self.smoothing_param)
        elif (self.smoothing == 'moving_average'):
          for i in range(len(smoothed_predictions)-1):
            unsmoothed_predictions.append(smoothed_predictions[i] * self.smoothing_param - sum(unsmoothed_actual[max(0, i-self.smoothing_param+1):i]))

        forecast_dates_2 = forecast_dates[:-1]
        df_forecast = pd.DataFrame({'Time':np.array(forecast_dates_2), 'Predicted Target':unsmoothed_predictions})

        df_forecast['Time'] = pd.to_datetime(df_forecast['Time'])

        # Center forecast (denormalize)
        mean_pred = df_forecast['Predicted Target'].mean()
        centered_pred = df_forecast['Predicted Target'] - mean_pred
        doubled_pred = centered_pred + mean_pred   
        df_forecast['Predicted Target'] = doubled_pred

        # Prepare Date column
        original = self.Y.to_frame()
        original['Time'] = pd.to_datetime(self.df_for_training.index)

        # Calculate the time interval between timesteps to shift (n_future-1) timesteps to the left (alignment)
        time_interval = df_forecast['Time'].diff().mean()
        df_forecast['Time'] = df_forecast['Time'] - (self.n_future-1) * time_interval
        original = original[:-(self.n_future)]

        # Slice the original DataFrame to get the last 2.5% of rows
        last_percent_index = int(len(original) * 0.975)
        last_percent_original = original.iloc[last_percent_index:].copy()

        # Plot the results
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=last_percent_original, x='Time', y=last_percent_original.iloc[:, 0], label='Actual')
        sns.lineplot(data=df_forecast, x='Time', y='Predicted Target', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Unsmoothed Time-series Prediction')
        plt.legend()
        plt.show()

        return df_forecast

    def save_trained_model(self, file_path: str) -> None:
        """
        Saves the trained model to the specified file path.
        
        Parameters:
        file_path (str): The path where the model will be saved.
        
        Returns:
        None
        """
        if self.model is not None:
            self.model.save(file_path)
            print(f"Model saved to {file_path}")
        else:
            raise ValueError("No model has been trained yet.")