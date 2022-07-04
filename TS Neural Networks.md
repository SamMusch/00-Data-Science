Resources

[Book: Deep Learning for Time Series Forecasting](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/) by Jason Brownlee

[Udacity: Time Series Forecasting w TensorFlow (Free)](https://learn.udacity.com/courses/ud187)

- Author: Aurélien Géron. He is the author of [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).

# Lectures

## 0. Basics Overview

- [4. Common patterns](https://learn.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/6630bdbc-2063-4010-83a4-6b61d1baebbc): White noise, trend, seasonality
- [6. Forecasting](https://learn.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/23defc4f-9b23-4335-84e4-6779d3cdd0b0): Naive forecast, fixed vs roll forward partitioning
- [8. Metrics](https://learn.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/1c065978-54ce-48cc-afb4-99113a9acf16): Differencing, MA, smoothing
- [10. Time Windows](https://learn.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/d4f22578-ee8f-4c2c-ad4f-c7421f5406e4)



Steps:

1. **Tuning**: Train on training data, test on validation data

2. **Estimating production**: Train on training & validation data, test on test data

3. **Production**: Train on all 3, predict out

   

## 01. Pre-Steps

We want to make the time series as simple as possible before sending it to the model.

Need to get rid of the following:

- Trend
- Seasonality (months, weekdays, etc)
  - Make sure train-val-test captures this seasonality

Use **roll-forward** partitioning instead of **fixed** partitioning  ([Video](https://learn.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/23defc4f-9b23-4335-84e4-6779d3cdd0b0))

- Fixed: Normal

- Roll forward: Start with a short training period and then predict out. (Essentially mimicking real-life).

  Note: Takes much longer



**Metrics video**

- **Differencing**: This helps get rid of the trend & seasonality

- **MA**: Eliminates some noise but does not anticipate trend & seasonality (apply differencing first)

- **Forecast for both** = 

  trailing MA of differencing TS + centered MA of past series (t-365)

```python
import pandas as pd
series = pd.Series(series)

split_time = 1000    # Train vs test
ts_diff = 365        # Number of time periods to use for differencing
ts_ma = 50           # Number of time periods to use for moving average
ts_smooth_past = 11
ts_smooth_begin = ts_diff + np.floor(ts_smooth_past / 2)
ts_smooth_end = ts_diff - np.ceil(ts_smooth_past / 2)

# Differencing
diff_series = series.diff(ts_diff).dropna()

# MA
diff_moving_avg = diff_series.rolling(ts_ma, closed='left').mean().dropna().iloc[split_time - ts_diff - ts_ma:]
diff_moving_avg_plus_past = (diff_moving_avg + series.shift(ts_diff)).dropna()

# Both
smoothed = series.rolling(ts_smooth_past, closed='left').mean().dropna().iloc[split_time - int(ts_smooth_begin):-int(ts_smooth_end)]
diff_moving_avg_plus_smooth_past = smoothed + diff_moving_avg.values
```



---

## 04. Windowing

The main features of the input windows are:

- The width (number of time steps) of the input and label windows.
- The time offset between them. 
- Which features are used as inputs, labels, or both. 



Example: Take 24 hours and give a prediction 24 hours in the future.

- Input width = 24
- Offset = 24
- Total width = 48
- Label width = 1



[Intro to Tensors](https://www.tensorflow.org/guide/tensor)  

- Tensor: Think of them as np.array that can be 1D, 2D, 3D, etc.
  - Can be 1 column or more, need to be the same dtype. Basically an np.array.


- Element: Each value in a tensor. Could be nested which would then contain multiple components 

---

## 05. ML

Video: 12. Forecasting with ML

[Sample, Batch, Epoch](https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean)

- **Sample**: one element of a dataset. (One row)

- **Batch**: a set of *N* samples. The larger the batch, the better the approximation; pick as large as you can afford without running out of memory

- **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation. 

SGD with some momentum helps converge quickly. Could try Adam as well.

Huber Loss for training: Good for optimizing MAE

- quadratic for small errors (MSE)
- linear for large errors (MAE)

Early Stopping Callback: 

- Patience = 10  --> Interupts training when validation doesn't improve for 10 consecutive epochs
- This allows us to set epochs = 500 because early stopping will happen way sooner

---

## 06. RNN

**Vector transfer**: conveyor belt with only some minor linear interactions

**Gates**: Outputs how much information to let through (0 to 1)

- **Neural Network Layer**: Yellow
- **Pointwise operator**: Pink

![image-20220102123401845](/Users/Sam/Library/Application Support/typora-user-images/image-20220102123401845.png)



[TensorFlow Guide on Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

RNN's are networks with loops in them, allowing information to persist.

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor.

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.

All recurrent neural networks have the form of a chain of repeating modules of neural network.



[14. Video](https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/cb52c9ee-1abd-48d0-818a-3f7342a7b079)

Process of RNN (RNN: Contains recurrent layers)

1. Take in the 3D **input windows**
   1. Batch size
   2. \# of time steps
   3. \# of features in the model
2. Send to a Recurrent Layer, composed of a single memory cell
   1. Take value from previous time step
   2. Output value for current time step **AND** the state/context so the model runs sequentially
   3. Repeat
3. Repeat \#2
4. Output forecast (ie Sequence to Vector)

<img src="/Users/Sam/Library/Application Support/typora-user-images/image-20220123092302104.png" alt="image-20220123092302104" style="zoom:33%;" />

---



### Things to be aware of

[Video 1](https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/29b5712f-4228-4503-a0e2-3a00a0cfd530)

1. Do I have the right number of neurons?
2. Do I have the right number of layers?
3. Learning rate too..
   1. High: Training will be unstable, model won't learn
   2. Low: Training will be slow
4. Do I have early stopping set right? Loss can jump up/down unpredictably during training.



[Video](https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/0132f4de-dbaf-4d30-b562-f6469c7da3a8)

1. Vanishing gradient: This often occurs when back propagating through many layers / time steps, especially when detecting long term patterns.
   1. 1 Approach: Make a prediction at each step time (ie Sequence to Sequence). Function: `seq2seq_window_dataset`
2. RNNs are useful when we have lots of high-frequency data and the signal:noise ratio is high



# Deep Learning Ebook

## 02. Taxonomy

1. **Inputs vs. Outputs** (X vs Y)
   - Inputs: Historical data provided to the model in order to make a single forecast.  
   - Outputs: Forecast for a future time step beyond the data provided as input.
2. **Endogenous vs. Exogenous** (Influencing each other?)
   - Endogenous: Input variables that *are* influenced by other variables in the system and on which the output variable depends.  
   - Exogenous: Input variables that *are not* influenced by other variables in the system and on which the output variable depends.
3. **Unstructured vs. Structured** (Time-dep patterns?)
   - Unstructured: No obvious systematic time-dependent pattern in a time series variable.  
   - Structured: Systematic time-dependent patterns in a time series variable (e.g. trend and/or seasonality).
4. **Univariate vs. Multivariate**
   - Uni and Multi Inputs: 1+ input variables measured over time.  
   - Uni and Multi Outputs: 1+ output variables to be predicted.
5. **Single-step vs. Multi-step**
   - One-step: Forecast the next time step.  
   - Multi-step: Forecast more than one future time steps.
6. **Static vs. Dynamic** (Streaming?)
   - Static: Model is fit once and used to make predictions.  
   - Dynamic: Model is fit on newly available data prior to each prediction.
7. **Contiguous vs. Discontiguous** (Time uniform?)
   - Contiguous: Observations are uniform over time.  (eg 1 per hour)
   - Discontiguous: Observations are not uniform over time.



## 04. Windows

Sliding window: Take all columns in the dataset (including target variable) and take the **lag**.

Parameters for the **lag**:

- **Input Width**: Number of time steps
- **Offset**: "1" if just using the values from previous time step
- Total width: Input Width + Offset
- **Label width**: How many timesteps in the future 



## 06. Data Transform

Input shape:

- **Samples:** One sequence is one sample. A batch is comprised of one or more samples.  
- **Time Steps:** One time step is one point of observation in the sample. One sample is comprised of multiple time steps.  
- **Features:** One feature is one observation at a time step. One time step is comprised of one or more features.

Put Simply:

- Normal Shape: Rows, Columns
- TS Shape: Rows, **TimeSteps**, Columns



# Extras

Consider:

- Fourier Transform
- Add exponential decay
- [Paper on LSTM being Unstable](https://arxiv.org/pdf/1805.10369.pdf)

Compare model against:

- ARIMA
- Prophet
- Naive (previous time period)
