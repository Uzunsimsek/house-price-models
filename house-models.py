# Doğrusal Olmayan Regresyon Modelleri
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
# !pip install catboost
from catboost import CatBoostRegressor

# !pip install lightgbm
# conda install -c conda-forge lightgbm
from lightgbm import LGBMRegressor

# !pip install xgboost
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

train_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
test_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")


# train_df tüm veri setimiz gibi davranarak derste ele aldığımız şekilde modelelme işlemini gerçekleştiriniz.
X = train_df.drop('SalePrice', axis=1)
y = train_df[["SalePrice"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)


# TODO scaler'i burada çalıştırıp deneyebilirsiniz.

models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)

#np.expm1(df["SalePrice"].mean())

X_train.shape, test_df.shape


train_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
test_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")
"Id" in test_df

# Bariz hatalı 2 değişkenin çıkarılması:
all_data = [train_df, test_df]
drop_list = ["index", "Id"]


for data in all_data:
    data.drop(drop_list, axis=1, inplace=True)

# train & test ayrımını yapalım:
X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[["SalePrice"]])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)  # boyut ayarlaması

# KNN: Model & Tahmin
knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Bu hatayı manuel olarak elde edip k hiperarametresinin değişimini gözlemleyelim:
RMSE = []


for k in range(20):
    k = k + 2
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k =", k, "için RMSE değeri:", rmse)

# GridSearchCV yöntemi ile optimum k'yı bulalım:
knn_params = {"n_neighbors": np.arange(2, 30, 1)}
knn_model = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_

# Final Model
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# SVR: Model & Tahmin
svr_model = SVR("linear").fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# SVR Tuning
svr_model = SVR("linear")
svr_params = {"C": [0.01, 0.1, 1, 10, 100]}
# svr_params2 = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_

# Final Model
svr_tuned = SVR("linear", C=100).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# NON-Linear SVR: Model & Tahmin
svr_model = SVR()
svr_params = {"C": [0.01, 0.1, 1, 10, 100]}
# svr_params2 = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_

# Final Model
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Yapay Sinir Ağları: Model & Tahmin
mlp_model = MLPRegressor().fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
mlp_params = {"alpha": [0.1, 0.01],
              "hidden_layer_sizes": [(10, 20), (5, 5)]}

# mlp_params2 = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001],
#               "hidden_layer_sizes": [(10, 20), (5, 5), (100, 100), (1000, 100, 10)],
#               "solver" : ['lbfgs', 'sgd', 'adam'],
#               "alpha": [10 ** np.linspace(10, -2, 100) * 0.5]}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
mlp_cv_model.best_params_

# Final Model
mlp_tuned = MLPRegressor(**mlp_cv_model.best_params_).fit(X_train, y_train)

y_pred = mlp_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))



# CART
cart_model = DecisionTreeRegressor(random_state=52)
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
cart_params = {"max_depth": [10, 20, 100],
               "min_samples_split": [15, 30, 50]}

# cart_params2 = {"max_depth": [2, 3, 4, 5, 10, 20, 100, 1000],
#               "min_samples_split": [2, 10, 5, 30, 50, 10],
#               "criterion" : ["mse", "friedman_mse", "mae"]}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10).fit(X_train, y_train)
cart_cv_model.best_params_

# Final Model
cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Karar Kuralları
# !pip install skompiler
# pip install astor
from skompiler import skompile

print(skompile(cart_tuned.predict).to('python/code'))


# Random Forests: Model & Tahmin
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
rf_params = {"max_depth": [5, 8, None],
             "max_features": [20, 50, 100],
             "n_estimators": [200, 500],
             "min_samples_split": [2, 5, 10]}

# rf_params2 = {"max_depth": [3, 5, 8, 10, 15, None],
#            "max_features": [5, 10, 15, 20, 50, 100],
#            "n_estimators": [200, 500, 1000],
#            "min_samples_split": [2, 5, 10, 20, 30, 50]}


rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_



# Final Model
rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Feature Importance

os.getcwd()

rf_tuned.feature_importances_
Importance = pd.DataFrame({'Importance': rf_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('rf_importance.png')


# GBM: Model & Tahmin
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# # Model Tuning
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

# gbm_params2 = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
#               "max_depth": [3, 5, 8, 10,20,30],
#               "n_estimators": [200, 500, 1000, 1500, 5000],
#               "subsample": [1, 0.4, 0.5, 0.7],
#               "loss": ["ls", "lad", "quantile"]}

gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model,
                            gbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_

# Final Model
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Feature Importance
Importance = pd.DataFrame({'Importance': gbm_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('gbm_importance.png')


# XGBoost: Model & Tahmin
xgb = XGBRegressor().fit(X_train, y_train)
y_pred = xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8],
              "n_estimators": [100, 1000],
              "colsample_bytree": [0.7, 1]}

# xgb_params2 = {"learning_rate": [0.1, 0.01, 0.5],
#              "max_depth": [5, 8, 15, 20],
#              "n_estimators": [100, 200, 500, 1000],
#              "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_

# Final Model
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Feature Importance
Importance = pd.DataFrame({'Importance': xgb_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('xgb_importances.png')


# LightGBM: Model & Tahmin
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

# lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
#               "n_estimators": [200, 500, 1000, 5000],
#               "max_depth": [6, 8, 10, 15, 20],
#               "colsample_bytree": [1, 0.8, 0.5, 0.4]}


lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Feature Importance
Importance = pd.DataFrame({'Importance': lgbm_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('lgbm_importances.png')


# CatBoost: Model & Tahmin
catb_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
catb_params = {"iterations": [200, 500],
               "learning_rate": [0.01, 0.1],
               "depth": [3, 6]}

catb_model = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb_model,
                             catb_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

catb_cv_model.best_params_

# Final Model
catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred))



Importance = pd.DataFrame({'Importance': catb_tuned.feature_importances_ * 100,
                           'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('catb_importances.png')

# 1. temel modeller ve hatalarına
# 2. model hiperparametre tuning
# 3. final modelleri
# 4. model optimizasyonu (feature engineer, aykırı, eksik, voting)


# Tum Base Modeller

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


# Base modellerin test hataları
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)

# Tüm veriye CV uygulayarak tüm base modellerin incelenmesi
for name, model in models:
    rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)


# Tune Edilmiş Modellerin Test Hataları:
tuned_models = [('KNN', knn_tuned),
                ('CART', cart_tuned),
                ('RF', rf_tuned),
                ('SVR', svr_tuned),
                ('GBM', gbm_tuned),
                ("XGBoost", xgb_tuned),
                ("LightGBM", lgbm_tuned),
                ("CatBoost", catb_tuned)]

for name, model in tuned_models:
    # model.fit(X_train, y_train) # tekrar fit etmeye gerek yok.
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)


# Tune Edilmiş Modellerin Kaydedilmesi
# models isminde bir klasör açıyorum.
# working directory'i kaydediyorum.
cur_dir = os.getcwd()
cur_dir
# directory'yi değiştiriyorum:
os.chdir('5.Hafta/Models')

for model in tuned_models:
    pickle.dump(model[1], open(str(model[0]) + ".pkl", 'wb'))

# kaydedilmiş bir modeli cagiralim:
rf = pickle.load(open('RF.pkl', 'rb'))
rf.predict(X_test)[0:5]

svr = pickle.load(open('SVR.pkl', 'rb'))
svr.predict(X_test)[0:5]


model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(test_df)
lgbm_pred = np.expm1(model.predict(test_df))

test3 = pd.read_csv("5.Hafta/Dataset/test.csv")
sub = pd.DataFrame()
sub['Id'] = test3.Id
sub['SalePrice'] = lgbm_pred
sub.to_csv('model_lgbm_predict.csv',index=False)




# LightGBM: Model & Tahmin
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

# lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
#               "n_estimators": [200, 500, 1000, 5000],
#               "max_depth": [6, 8, 10, 15, 20],
#               "colsample_bytree": [1, 0.8, 0.5, 0.4]}


lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))



# KNN: (0.154828)
# CART: (0.193278)
# RF: (0.122072)
# SVR: (0.105167)
# GBM: (0.114463)
# XGBoost: (0.113279)
# LightGBM: (0.109378)
# CatBoost: (0.106999)













