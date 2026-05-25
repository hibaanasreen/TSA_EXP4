# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 25/05/26



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# -------------------- LOAD DATA -------------------- #

# Correct dataset path
file_path = "student_performance.csv"

# Read dataset
data = pd.read_csv(file_path)

# -------------------- DISPLAY COLUMNS -------------------- #

print("Columns in Dataset:")
print(data.columns)

# -------------------- SELECT NUMERIC COLUMN -------------------- #

numeric_cols = data.select_dtypes(include=[np.number])

if numeric_cols.empty:
    print("No numeric columns found in dataset.")

else:
    
    # Select first numeric column
    column_name = numeric_cols.columns[0]

    # Remove missing values
    X = numeric_cols[column_name].dropna()

    print("\nSelected Column for Time Series:", column_name)

    # -------------------- VISUALIZE DATA -------------------- #

    plt.figure(figsize=(12, 6))

    plt.plot(X)

    plt.title(f'Time Series Plot of {column_name}')

    plt.xlabel('Time')

    plt.ylabel('Values')

    plt.grid(True)

    plt.show()

    # -------------------- CHECK STATIONARITY -------------------- #

    result = adfuller(X)

    print("\nADF Statistic:", result[0])

    print("p-value:", result[1])

    # Apply differencing if non-stationary
    if result[1] > 0.05:

        print("\nData is NOT stationary → Applying Differencing")

        X = X.diff().dropna()

    else:

        print("\nData is Stationary")

    # -------------------- ACF & PACF -------------------- #

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)

    plot_acf(X, lags=min(30, len(X)//2), ax=plt.gca())

    plt.title('Autocorrelation Function (ACF)')

    plt.subplot(2, 1, 2)

    plot_pacf(X, lags=min(30, len(X)//2), ax=plt.gca())

    plt.title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()

    plt.show()

    # -------------------- ARMA(1,1) MODEL -------------------- #

    print("\nFitting ARMA(1,1) Model...")

    arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

    phi1 = arma11_model.params.get('ar.L1', 0)

    theta1 = arma11_model.params.get('ma.L1', 0)

    print("\nARMA(1,1) Parameters")

    print("phi =", phi1)

    print("theta =", theta1)

    # Simulate ARMA(1,1)

    N = 500

    ar1 = np.array([1, -phi1])

    ma1 = np.array([1, theta1])

    simulated_arma11 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

    plt.figure(figsize=(12, 5))

    plt.plot(simulated_arma11)

    plt.title('Simulated ARMA(1,1)')

    plt.grid(True)

    plt.show()

    plot_acf(simulated_arma11)

    plt.title('ACF of Simulated ARMA(1,1)')

    plt.show()

    plot_pacf(simulated_arma11)

    plt.title('PACF of Simulated ARMA(1,1)')

    plt.show()

    # -------------------- ARMA(2,2) MODEL -------------------- #

    print("\nFitting ARMA(2,2) Model...")

    arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

    phi1 = arma22_model.params.get('ar.L1', 0)

    phi2 = arma22_model.params.get('ar.L2', 0)

    theta1 = arma22_model.params.get('ma.L1', 0)

    theta2 = arma22_model.params.get('ma.L2', 0)

    print("\nARMA(2,2) Parameters")

    print("phi1 =", phi1)

    print("phi2 =", phi2)

    print("theta1 =", theta1)

    print("theta2 =", theta2)

    # Simulate ARMA(2,2)

    ar2 = np.array([1, -phi1, -phi2])

    ma2 = np.array([1, theta1, theta2])

    simulated_arma22 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

    plt.figure(figsize=(12, 5))

    plt.plot(simulated_arma22)

    plt.title('Simulated ARMA(2,2)')

    plt.grid(True)

    plt.show()

    plot_acf(simulated_arma22)

    plt.title('ACF of Simulated ARMA(2,2)')

    plt.show()

    plot_pacf(simulated_arma22)

    plt.title('PACF of Simulated ARMA(2,2)')

    plt.show()

```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:


<img width="643" height="350" alt="image" src="https://github.com/user-attachments/assets/edfc7b45-ac13-46d8-8d84-e514d66b0c8d" />



Partial Autocorrelation ,Autocorrelation:

<img width="1116" height="553" alt="image" src="https://github.com/user-attachments/assets/c1271471-478a-4724-b243-300d1e7ad687" />


SIMULATED ARMA(1,1) PROCESS:

<img width="913" height="420" alt="image" src="https://github.com/user-attachments/assets/9794b9c6-9f5d-4ad2-86e4-ca41a177d612" />



<img width="360" height="560" alt="image" src="https://github.com/user-attachments/assets/6595ee89-7de0-4775-9963-f6bcfd443877" />


SIMULATED ARMA(2,2) PROCESS:

<img width="615" height="282" alt="image" src="https://github.com/user-attachments/assets/98fa9eea-5662-4830-942f-f625f8af2043" />

<img width="362" height="554" alt="image" src="https://github.com/user-attachments/assets/2a7907a3-2983-4035-9f61-db2ab9ca1fc2" />


Partial Autocorrelation

<img width="745" height="187" alt="image" src="https://github.com/user-attachments/assets/7468d2d2-a2c2-46fd-9b63-413cbacb289b" />


Autocorrelation

<img width="752" height="185" alt="image" src="https://github.com/user-attachments/assets/ed921afa-064c-44fd-a0f2-fb02cd4387eb" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
