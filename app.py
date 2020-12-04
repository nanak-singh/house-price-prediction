import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
from statsmodels.api import OLS
import numpy as np
# r2 score, rms ,mae usings sklearn
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import sklearn.linear_model as lm

data = pd.read_csv("https://trello-attachments.s3.amazonaws.com/5cf2142046ceb163a0e4b189/5cf4e21e143159856a320b36/5311d0b467c098f8f83a90a6da607751/Housing_Modified_prepared(1).csv")

def main():
    
    title_html = """
    
        <style>

            .title h1 {
                text-align: center;
                font-size: 50px;
                color: #FEE469;
            }
            
            body {
                background-color: #54B9C1;
            }

        </style>
        <body>
        <div class="title">
            <h1><i>Industrial Training Project</i></h1>
        </div>

    """

    st.markdown(title_html, unsafe_allow_html=True)


    FiveLayerHashing = """
    
        <style>

            .title h3 {
                text-align: center;
                font-size: 35px;
                color: red;

            }

        </style>

        <div class="title">
            <h3><i>House Price Prediction Model</i></h3>
        </div>

    """
    st.markdown(FiveLayerHashing, unsafe_allow_html=True)
    #st.title("House Price Prediction Model")
    #st.sidebar.title("House Price Prediction")

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'LUVPREET SINGH  | CSE-1 | 05713202718'; 
                visibility: visible;
                display: block;
                text-align: center;
                position: relative;
                color: black;
                #background-color: red;
                font-size: 20px;
                padding: 5px;
                padding-tip: -50px;
                top: 2px;
            }

            </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    #x = data["lotsize"]
    #xMean = x.mean()
    #xStd = x.std()
    #xNorm = (x - xMean) / xStd
    independentVariables = data.columns
    independentVariables = independentVariables.delete(0)
    #x = data[independentVariables]
    #y = data["price"]
    #scale = StandardScaler
    #x = data["lotsize"]
    #xMin = x.min()
    #xMax = x.max()
    #xNorm = (x - xMin) / (xMax - xMin)

    #x = data[independentVariables]
    #xMin = x.min()
    #xMax = x.max()
    #xNorm = (x - xMin) / (xMax - xMin)
    #scale = MinMaxScaler()
    #xNorm = scale.fit_transform(x)
    y = data["price"]
    independentVar = data.columns
    independentVar = independentVar.delete(0)
    x = data[independentVar]
    lr = LinearRegression()
    lr.fit(x,y)
    OLS(y,x).fit()
    #ypred = lr.predict(x)
    #ymean = y.mean()
    #sqTotal = np.square(y - ymean)
    #sst = sqTotal.sum()
    #ssr(sum of sq resideual)
    #squaredResideual = np.square(ypred-ymean)
    #ssr = squaredResideual.sum()
    #calc rsquare = r2score
    #r2Score = (ssr / sst)
    # calc rms error
    #error = y - ypred
    #sqError = np.square(error)

    #mean
    #sse = sqError.sum()
    #meanError = sse / len(y)
    #rmse = np.sqrt(meanError)


    #mean absolute error
    #absError = abs(y - ypred)
    #sae = absError.sum()
    #mae = sae / len(y)

    #r2score = r2_score(y,  ypred)

    y = data["price"]
    independentVar = data.columns
    independentVar = independentVar.delete(0)
    x = data[independentVar]

    model = sm.OLS(y,x)
    model = model.fit()

    for i in range(len(independentVar)):
        vif_list = [vif(data[independentVar].values,index) for index in range(len(independentVar))]
        mvif = max(vif_list)
        dropIndex = vif_list.index(mvif)

        if mvif>10:
            independentVar = independentVar.delete(dropIndex)

    
    y = data["price"]
    x = data[independentVar]
    model = sm.OLS(y,x)
    model = model.fit()


    user_input = { 
        "lotsize":[], 
        "bathrms":[], 
        "stories":[], 
        "driveway":[], 
        "recroom":[], 
        "fullbase":[], 
        "gashw":[], 
        "airco":[], 
        "garagepl":[], 
        "prefarea":[] 
    }

    #aa = st.number_input("Enter Lotsize:", min_value=0, step=1)
    aa = st.number_input("Enter Lotsize:", min_value=0.00)
    user_input["lotsize"].append(aa)

    bb = int(st.number_input("Enter BathRooms:", min_value=0.00))
    user_input["bathrms"].append(bb)
    
    cc = int(st.number_input("Enter Stories:", min_value=0.00))
    user_input["stories"].append(cc)
    
    dd = int(st.number_input("Enter DriveWay:", min_value=0.00))
    user_input["driveway"].append(dd)
    
    ee = int(st.number_input("Enter RecRoom:", min_value=0.00))
    user_input["recroom"].append(ee)
    
    ff = int(st.number_input("Enter FullBase:", min_value=0.00))
    user_input["fullbase"].append(ff)
    
    gg = int(st.number_input("Enter GasHW:", min_value=0.00))
    user_input["gashw"].append(gg)
    
    hh = int(st.number_input("Enter AirCo:", min_value=0.00))
    user_input["airco"].append(hh)
    
    ii = int(st.number_input("Enter GaragePl:", min_value=0.00))
    user_input["garagepl"].append(ii)
    
    jj = int(st.number_input("Enter PrefArea:", min_value=0.00))
    user_input["prefarea"].append(jj)

    user_df = pd.DataFrame(data = user_input, index=[0], columns = independentVar)

    #lr = lm.LinearRegression()
    #lr.fit(x,y)
    


    if st.button("PREDICT"):
        price = ''
        lr = lm.LinearRegression()
        lr.fit(x,y)
        price = lr.predict(user_df)
        if aa == 0.0 and bb == 0.0 and cc == 0.0 and dd == 0.0 and ee == 0.0 and ff == 0.0 and gg == 0.0 and hh == 0.0 and ii == 0.0 and jj == 0.0 :
            price[0] = 0
        st.markdown("House Price in USD = " + str(int(abs(price[0]))))

    body_ends = """
        </body>
    """

    st.markdown(body_ends, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
