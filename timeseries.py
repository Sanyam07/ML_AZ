# time series RNN

from util import Util
import tensorflow as tf
from tensorflow import keras

batch_size = 10000
n_steps = 50
series = Util.generate_time_series(batch_size, n_steps+1)

X_train, y_train = series[:7000,:n_steps], series[:7000, n_steps:]
X_validate, y_validate = series[7000:9000,:n_steps], series[7000:9000, n_steps:]
X_test, y_test = series[9000:,:n_steps], series[9000:, n_steps:]


#### base metric-1
y_pred = X_validate[:,-1]
mse = keras.metrics.mean_squared_error(np.squeeze(y_validate), np.squeeze(y_pred))

#### base metric-2
model= keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(50,1)))
model.add(keras.layers.Dense(1))
model.compile(loss=keras.metrics.mse)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_validate,y_validate))

#### RNN
model2 = keras.models.Sequential()
model2.add(keras.layers.SimpleRNN(1, input_shape=(None,1)))
model2.compile(loss=keras.metrics.mean_squared_error)
history2 = model2.fit(X_train, y_train, epochs=20, validation_data=(X_validate, y_validate))

model3 = keras.models.Sequential()
model3.add(keras.layers.SimpleRNN(20, return_sequences=True,  input_shape=(None,1)))
model3.add(keras.layers.SimpleRNN(20, return_sequences=True))
model3.add(keras.layers.SimpleRNN(1))
model3.compile(loss = keras.metrics.mse)
hitory3 = model3.fit(X_train, y_train, epochs=20, validation_data=(X_validate,y_validate))

model3 = keras.models.Sequential()
model3.add(keras.layers.SimpleRNN(20, return_sequences=True,  input_shape=(None,1)))
model3.add(keras.layers.SimpleRNN(20))
model3.add(keras.layers.Dense(1))
model3.compile(loss = keras.metrics.mse)
hitory3 = model3.fit(X_train, y_train, epochs=20, validation_data=(X_validate,y_validate))