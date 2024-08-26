The `DesmoothForecasterModel` class provides a framework for time-series forecasting using a method that allows for original future timestep predictions to be made based on predictions on the smoothed version of the data. Currently, the model uses a simple LSTM with an option to add a custom model. It applies smoothing techniques to the target variable before training the provided model on predicting smoothed values, and then extrapolates unsmoothed, original predictions from them. 

## Installation

To install the `DesmoothForecaster` package, you can simply run the following command in your terminal or command prompt:

```bash
pip install DesmoothForecaster
```

This will download and install the package along with its dependencies. The package is compatible with Python 3.6 and above, and works on windows, linux, and macos.

## Class: `DesmoothForecasterModel`

### Description
A class to create and train a model that uses smoothed time-series data for forecasting. The model supports LSTM or custom neural network architectures and can apply exponential smoothing or moving average smoothing to the target variable before training.

### Attributes

- **`model_type`** (`str`): The type of model to use (`'LSTM'` or `'custom'`). If custom, a custom model must be provided.
- **`custom_model`** (`keras.Model`, optional): A custom model to use instead of the default LSTM model. model_type must be set to `'custom'`.
- **`smoothing`** (`str`): The smoothing technique to apply (`'exp_smoothing'` or `'moving_average'`).
- **`smoothing_param`** (`float` or `int`): The parameter for the chosen smoothing method. For exponential smoothing, this is the smoothing level (alpha). For moving average, this is the window size.
- **`scaler`** (`StandardScaler`): A scaler instance used to normalize the data.
- **`model`** (`keras.Model`, optional): The neural network model to be trained.
- **`trainX`** (`np.ndarray`, optional): The input training data prepared for the model.
- **`df_for_training`** (`pd.DataFrame`, optional): The DataFrame containing smoothed and normalized data used for training.
- **`n_future`** (`int`, optional): The number of future timesteps the model is trained to predict.
- **`Y`** (`pd.Series`, optional): The original target variable before smoothing.
- **`lookback`** (`int`, optional): The number of past timesteps used to predict the future.

### Methods

#### `__init__(model='LSTM', custom_model=None, smoothing="exp_smoothing", smoothing_param=-1)`

**Description**:  
Initializes the `DesmoothForecasterModel` with the specified parameters.

**Parameters**:  
- **`model`** (`str`, optional): The type of model to use (`'LSTM'` by default, or `'custom'`).
- **`custom_model`** (`keras.Model`, optional): A custom model to use instead of the default LSTM model (default is `None`).
- **`smoothing`** (`str`, optional): The smoothing technique to apply (`'exp_smoothing'` by default, or `'moving_average'`).
- **`smoothing_param`** (`float` or `int`, optional): The parameter for the smoothing method. Default is `-1`, which auto-sets the value based on the smoothing method: `0.01` for exponential smoothing and `10` for moving average.

#### `smooth_data(Y) -> pd.Series`

**Description**:  
Applies the chosen smoothing technique to the target variable.

**Parameters**:  
- **`Y`** (`pd.Series`): The target variable to be smoothed.

**Returns**:  
- **`pd.Series`**: The smoothed target variable.

#### `prepare_data(time_column, X, Y, lookback, n_future) -> np.ndarray`

**Description**:  
Prepares the input and output data for training the model by applying smoothing, normalizing, and reshaping it into sequences.

**Parameters**:  
- **`time_column`** (`pd.Series` or `pd.Index`): The time indices or datetime values for the data.
- **`X`** (`pd.DataFrame`): The input feature columns.
- **`Y`** (`pd.Series`): The target variable column.
- **`lookback`** (`int`): The number of past timesteps used to predict the future.
- **`n_future`** (`int`): The number of future timesteps to predict.

**Returns**:  
- **`np.ndarray`**: The prepared output data (target variable) for training.

#### `build_model(input_shape) -> Sequential`

**Description**:  
Builds the neural network model based on the specified model type.

**Parameters**:  
- **`input_shape`** (`tuple`): The shape of the input data for the model.

**Returns**:  
- **`keras.Model`**: The compiled neural network model.

#### `train_model(time_column, X, Y, lookback, n_future, epochs, batch_size=32, validation_split=0.2) -> Sequential`

**Description**:  
Trains the neural network model on the prepared data.

**Parameters**:  
- **`time_column`** (`pd.Series` or `pd.Index`): The time indices or datetime values for the data.
- **`X`** (`np.ndarray`): The input data for training, prepared by the `prepare_data` method.
- **`Y`** (`np.ndarray`): The target variable for training.
- **`lookback`** (`int`): The number of past timesteps used to predict the future.
- **`n_future`** (`int`): The number of future timesteps to predict.
- **`epochs`** (`int`, optional): The number of epochs to train the model (default is `100`).
- **`batch_size`** (`int`, optional): The batch size used during training (default is `32`).
- **`validation_split`** (`float`, optional): The proportion of data to use for validation (default is `0.2`).

**Returns**:  
- **`keras.Model`**: The trained neural network model.

#### `predict(n_timesteps) -> pd.DataFrame`

**Description**:  
Predicts future values based on the trained model and prepared data.

**Parameters**:  
- **'data'** (`pd.DataFrame` or `None`, optional): The DataFrame containing the prepared data for prediction. If not provided, the last n_timesteps of the training data will be used.
- **`n_timesteps`** (`int`, optional): The number of future timesteps to predict.

**Returns**:  
- **`pd.DataFrame`**: A DataFrame containing the actual and predicted values with the corresponding time indices.

#### `save_trained_model(file_path: str) -> None`

**Description**:  
Saves the trained neural network model to the specified file path. This function can be used to persist the model after training so that it can be loaded and used later without retraining.

**Parameters**:  
- **`file_path`** (`str`): The file path where the trained model will be saved. This should include the file name and appropriate extension (e.g., `.h5` for Keras models).

**Returns**:  
- **`None`**
  
## Usage
To use the `DesmoothForecaster` package, you can import the `DesmoothForecasterModel` class and create an instance of it. Here's a simple example with pdw data from radars (loaded from google drive):
```python
import gdown
import pandas as pd
from DesmoothForecaster import DesmoothForecasterModel
file_id = '1-3DGmcR8PP9k_HSMmH8B7Lu8XK8vPyKF'
download_url = f'https://drive.google.com/uc?id={file_id}'
output = 'file.csv'
gdown.download(download_url, output, quiet=False)

df = pd.read_csv(output)
```

Create an instance of the `DesmoothForecasterModel` class with the desired configuration:
```python
features = ['PPB', 'APW', 'SNR']
n_future = 7
desmooth_instance = DesmoothForecasterModel(
    model_type='LSTM',          # Model type: 'LSTM' or 'custom'
    custom_model=None,          # Provide a custom model if using 'custom'
    smoothing="exp_smoothing",  # Smoothing method: 'exp_smoothing' or 'moving_average'
    smoothing_param=-1          # Smoothing parameter: set to -1 for automatic selection
)
```

Train the model using the `train_model` method:
```python
model = desmooth_instance.train_model(
    X=df.index,               # Index or time column
    y=df['PW'],               # Target variable (in this case, 'PW')
    features=df[features],    # Feature columns
    lookback=100,             # Number of past timesteps to consider
    n_future=n_future,        # Number of future timesteps to predict
    epochs=7                  # Number of training epochs
)
```

Use the trained model to predict future values and save it in the current directory.
```python
predictions = desmooth_instance.predict(n_future * 6)  # Predicting 6 times the future steps
desmooth_instance.save_trained_model('desmooth_model.h5')
```

## Contact
If you have any questions or feedback, please feel free to contact me at shandixit2002@gmail.com.
