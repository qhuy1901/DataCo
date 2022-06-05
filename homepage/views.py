from urllib import response
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from datetime import datetime, timedelta
import pandas as pd
from sklearn import preprocessing
import joblib
import os.path
import numpy as np

def show_result(request):
    response = HttpResponse()
    Order_City = request.GET.get("Order_City", None)
    Order_Region = request.GET.get("Order_Region", None)
    Order_Country = request.GET.get("Order_Country", None)
    Delivery_Status = request.GET.get("Delivery_Status", None)
    Order_Date = datetime.strptime(request.GET.get("Order_Date", None), '%Y-%m-%d')

    Shipping_Mode = request.GET.get("Shipping_Mode", None)
    Delivery_Date = datetime.strptime(request.GET.get("Delivery_Date", None), '%Y-%m-%d')
    Customer_State = request.GET.get("Customer_State", None)
    Customer_City = request.GET.get("Customer_City", None)
    Order_State = request.GET.get("Order_State", None)
    Payment_Method = request.GET.get("Payment_Method", None)

    df = pd.read_csv('https://raw.githubusercontent.com/qhuy1901/DataMining_DataCoSupplyChainDataset/main/dataco_supply_chain_processed_data_without_encode.csv', header= 0, encoding='unicode_escape')

    df.drop(['Real_Days_Shipping','Scheduled_Days_Shipping'], axis='columns', inplace=True)

    data = [{'Payment_Method': Payment_Method, 'Shipping_Mode':Shipping_Mode, 'Order_Region':Order_Region,
            'Order_Country':Order_Country, 'Order_City':Order_City, 'Delivery_Status': Delivery_Status,
            'Customer_City':Customer_City, 'Customer_State':Customer_State, 'Order_State': Order_State,'Order_Day':Order_Date.day,
        'Order_Month':Order_Date.month,'Order_Year':Order_Date.year,'Shipping_Day':Delivery_Date.day,'Shipping_Month':Delivery_Date.month,'Shipping_Year':Delivery_Date.year, 'Late_Delivery_Risk':0}]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)


    # Encode
    le = preprocessing.LabelEncoder()

    df['Payment_Method'] = le.fit_transform(df['Payment_Method'])
    df['Shipping_Mode'] = le.fit_transform(df['Shipping_Mode'])
    df['Order_Region'] = le.fit_transform(df['Order_Region'])
    df['Order_Country'] = le.fit_transform(df['Order_Country'])
    df['Order_City'] = le.fit_transform(df['Order_City'])
    df['Delivery_Status'] = le.fit_transform(df['Delivery_Status'])
    df['Customer_City'] = le.fit_transform(df['Customer_City'])
    df['Customer_State'] = le.fit_transform(df['Customer_State'])
    df['Order_State'] = le.fit_transform(df['Order_State'])

    df = df.tail(1)
    
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    Real_Days_Shipping_GNB_model = joblib.load(os.path.join(BASE, "Real_Days_Shipping_GNB_model.pkl"))
    Scheduled_Days_Shipping_GNB_model = joblib.load(os.path.join(BASE, "Scheduled_Days_Shipping_GNB_model.pkl"))

    # Use the loaded model to make predictions
    Real_Days_Shipping_Prediction = Real_Days_Shipping_GNB_model.predict(df)
    Scheduled_Days_Shipping_Prediction = Scheduled_Days_Shipping_GNB_model.predict(df)
    bayes_pred = pd.concat([pd.Series(Real_Days_Shipping_Prediction), pd.Series(Scheduled_Days_Shipping_Prediction)], axis = 1)

    Prediction = pd.DataFrame(bayes_pred)
    prediction = Prediction.rename(columns={0:'Fastest_shipment',1:'Avg_shipment'})

    prediction['risk'] = np.where(prediction['Avg_shipment'] >= prediction['Fastest_shipment'],0,1)

    result = ""
    for i in range(len(prediction)) :
        Fastest_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Fastest_shipment"]))
        Avg_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Avg_shipment"]))
        risk = "NO"
        if(int(prediction.loc[i, "risk"]) == 1):
            risk = "YES"

        result = """<div style="margin-bottom:20px;  text-align: center;"><b>Prediction result</b></div>
                                        <table class="table table-bordered table-head-bg-info table-bordered-bd-info">
                                            <thead>
                                                <tr>
                                                    <th scope="col">Fastest shipment</th>
                                                    <th scope="col">Scheduled shipment</th>
                                                    <th scope="col">Fastest receive goods date</th>
                                                    <th scope="col">Scheduled receive goods date</th>
                                                    <th scope="col">Late Delivery Risk</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td>{}</td>
                                                    <td>{}</td>
                                                    <td>{}</td>
                                                    <td>{}</td>
                                                    <td>{}</td>
                                                </tr>
                                            </tbody>
                                        </table>""".format(prediction.loc[i, "Fastest_shipment"], prediction.loc[i, "Avg_shipment"], Fastest_shipment_date.strftime("%b %d, %Y"), Avg_shipment_date.strftime("%b %d, %Y"), risk)

    response.write(result)

    return response


class HomeView(View):
    def get(seft, request):
        return render(request, 'homepage/forms.html')

 
