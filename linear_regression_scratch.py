import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


class sLinearRegression:
    def __init__(self, learning_rate=0.01, bias=0, n_iters=1000):
        # h(x) = bias + weights_1*x_1 + ... + weight_n*x_n
        self.lr=learning_rate
        self.weights=None
        self.bias=bias
        self.n_iters=n_iters

    def _init_param(self):
        self.weights=np.zeros(self.n_features)
        self.bias=0

    def fit(self,X,y):
        self.n_samples, self.n_features=X.shape
        self._init_param()
        for i in range(self.n_iters):
            y_pred=self.y_prediction(X)
            dweights,dbias=self.gradient_descent(X,y,y_pred)
            self.weights-= self.lr*dweights
            self.bias-=self.lr*dbias

    def gradient_descent(self, X, y, y_pred):
        diff=y_pred-y
        dweights=(1/self.n_samples)*np.dot(X.T, diff)
        dbias=(1/self.n_samples)*np.sum(diff)
        return dweights, dbias

    def predict(self,X):
        return self.y_prediction(X)
    
    def y_prediction(self, X):
        return self.bias+np.dot(X, self.weights)

# Testing from-scratch model
# Using fetch_california_housing dataset
X,y=fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

standardized=StandardScaler().fit(X_train)
sX_train = standardized.transform(X_train)
sX_test = standardized.transform(X_test)

slr = sLinearRegression()
slr.fit(sX_train,y_train)
spredict = slr.predict(sX_test)
s_mse = mean_squared_error(y_test, spredict)
s_rmse=s_mse**0.5
s_r2 = r2_score(y_test, spredict) # 1 is perfect fit, 0 is bad
print(f"FROM-SCRATCH MODEL:\nCaliforniaHousing dataset\nRMSE: {s_rmse}\nr2_score: {s_r2}")

# Using load_diabetes dataset
X2,y2=load_diabetes(return_X_y=True)
slr2 = sLinearRegression()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.2,random_state=1)
slr2.fit(X2_train,y2_train)
spredict = slr2.predict(X2_test)
s_mse = mean_squared_error(y2_test,spredict)
s_rmse = s_mse**0.5
s_r2=r2_score(y2_test, spredict)
print(f"\nDiabetes dataset\nRMSE: {s_rmse}\nr2_score: {s_r2}")

# Using make_regression dataset
X3,y3 = make_regression(n_samples=100, n_features=2,noise=10,random_state=42)
slr3 = sLinearRegression()
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.2,random_state=1)
slr3.fit(X3_train,y3_train)
spredict = slr3.predict(X3_test)
s_mse=mean_squared_error(y3_test,spredict)
s_rmse = s_mse**0.5
s_r2=r2_score(y3_test,spredict)
print(f"\nmake_regression dataset\nRMSE: {s_rmse}\nr2_score: {s_r2}")

# Testing on sklearn model
# Using fetch_california_housing dataset
lr = LinearRegression().fit(sX_train,y_train)
predict = lr.predict(sX_test)
mse=mean_squared_error(y_test, predict)
rmse=mse**0.5
r2=r2_score(y_test,predict)
print(f"\nSKLEARN MODEL:\nCaliforniaHousing dataset\nRMSE: {rmse}\nr2_score: {r2}")

# Using load_diabetes dataset
lr2 = LinearRegression().fit(X2_train,y2_train)
predict = lr2.predict(X2_test)
mse = mean_squared_error(y2_test, predict)
rmse = mse**0.5
r2 = r2_score(y2_test, predict)
print(f"\nDiabetes dataset\nRMSE: {rmse}\nr2_score: {r2}")

# Using make_regression dataset
lr3 = LinearRegression().fit(X3_train,y3_train)
predict = lr3.predict(X3_test)
mse = mean_squared_error(y3_test, predict)
rmse=mse**0.5
r2=r2_score(y3_test,predict)
print(f"\nmake_regression dataset\nRMSE: {rmse}\nr2_score: {r2}")