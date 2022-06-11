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
from sklearn.metrics import accuracy_score

def show_result(request):
    BASE = os.path.dirname(os.path.abspath(__file__))

    # Nhận data từ view 
    response = HttpResponse()
    showMultipleAlgorithms = request.GET.get("showMultipleAlgorithms", None)
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


    # Xử lý endcode dữ liệu raw của người dùng
    df = pd.read_csv(os.path.join(BASE, 'dataco_supply_chain_processed_data_without_encode.csv'), header= 0, encoding='unicode_escape')

    df.drop(['Real_Days_Shipping','Scheduled_Days_Shipping'], axis='columns', inplace=True)

    data = [{'Payment_Method': Payment_Method, 'Shipping_Mode':Shipping_Mode, 'Order_Region':Order_Region,
            'Order_Country':Order_Country, 'Order_City':Order_City, 'Delivery_Status': Delivery_Status,
            'Customer_City':Customer_City, 'Customer_State':Customer_State, 'Order_State': Order_State,'Order_Day':Order_Date.day,
        'Order_Month':Order_Date.month,'Order_Year':Order_Date.year,'Shipping_Day':Delivery_Date.day,'Shipping_Month':Delivery_Date.month,'Shipping_Year':Delivery_Date.year, 'Late_Delivery_Risk':0}]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    df = encode_data(df)
    df = df.tail(1)
    
    # Dự đoán
    prediction = predict(df, BASE)

    # In ra kết quả
    if(showMultipleAlgorithms == "1"):
        result = print_result_with_multiple_algorithms(prediction, Delivery_Date)
    else:
        result = print_result_with_ID3_algorithm(prediction, Delivery_Date)
    response.write(result)

    return response

def predict(df, BASE):
    # Get model form .pkl file
    Real_Days_Shipping_GNB_model = joblib.load(os.path.join(BASE, "Real_Days_Shipping_GNB_model.pkl"))
    Scheduled_Days_Shipping_GNB_model = joblib.load(os.path.join(BASE, "Scheduled_Days_Shipping_GNB_model.pkl"))

    LR_Real_Days_Shipping_model = joblib.load(os.path.join(BASE, "Real_Days_Shipping_LR_model.pkl"))
    LR_Scheduled_Days_Shipping_model = joblib.load(os.path.join(BASE, "Scheduled_Days_Shipping_LR_model.pkl"))

    CART_prediction_model = joblib.load(os.path.join(BASE, "CART_prediction_model.pkl"))

    ID3_prediction_model = joblib.load(os.path.join(BASE, "ID3_prediction_model.pkl"))

    # Use the loaded model to make predictions
    Real_Days_Shipping_Prediction = Real_Days_Shipping_GNB_model.predict(df)
    Scheduled_Days_Shipping_Prediction = Scheduled_Days_Shipping_GNB_model.predict(df)
    bayes_pred = pd.concat([pd.Series(Real_Days_Shipping_Prediction), pd.Series(Scheduled_Days_Shipping_Prediction)], axis = 1)

    lr_real_days_shipping_prediction = LR_Real_Days_Shipping_model.predict(df)
    lr_scheduled_days_shipping_prediction = LR_Scheduled_Days_Shipping_model.predict(df)
    lr_pred = pd.concat([pd.Series(lr_real_days_shipping_prediction), pd.Series(lr_scheduled_days_shipping_prediction)], axis = 1)

    cart_pred = pd.DataFrame(CART_prediction_model.predict(df))
    id3_pred = pd.DataFrame(ID3_prediction_model.predict(df))

    pred = pd.concat([id3_pred, cart_pred, bayes_pred, lr_pred], axis = 0)

    Prediction = pd.DataFrame(pred)
    prediction = Prediction.rename(columns={0:'Fastest_shipment',1:'Avg_shipment'})

    prediction['risk'] = np.where(prediction['Avg_shipment'] >= prediction['Fastest_shipment'],0,1)

    # Add algorithm_name column
    algorithm = ['ID3', 'CART', 'Naïve Bayes', 'Logistic Regression']
    accuracy_score = ['97.65%', '97.65%', '94.95%', '93.21%']
    r_square = ['0.993', '0.992', '0.861', '0.798']
    mse = ['0.01822', '0.01987', '0.36697', '0.50874']
    rmse = ['0.01468', '0.01556', '0.25449', '0.30948']
    prediction = prediction.assign(algorithm_name=algorithm, accuracy_score=accuracy_score,  r_square= r_square, mse=mse, rmse=rmse)

    prediction = prediction.reset_index(drop=True)
    return prediction

def encode_data(df):
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

    return df

def print_result_with_multiple_algorithms(prediction, Delivery_Date):
    result = """<div style="margin-bottom:20px;  text-align: center;"><b>Prediction result</b></div>
                                        <table class="table table-bordered table-head-bg-info table-bordered-bd-info">
                                            <thead>
                                                <tr>
                                                    <th scope="col" style="WIDTH: 180px">Algorithm</th>
                                                    <th scope="col">R square</th>
                                                    <th scope="col">MSE score</th>
                                                    <th scope="col">RMSE score</th>
                                                    <th scope="col">Fastest shipment</th>
                                                    <th scope="col">Scheduled shipment</th>
                                                    <th scope="col">Fastest received goods date</th>
                                                    <th scope="col">Scheduled received goods date</th>
                                                    <th scope="col" style="WIDTH: 170px">Late Delivery Risk</th>
                                                </tr>
                                            </thead>
                                            <tbody>"""
    for i in range(len(prediction)):
        Fastest_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Fastest_shipment"]))
        Avg_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Avg_shipment"]))
        risk = "NO"
        if(int(prediction.loc[i, "risk"]) == 1):
            risk = "YES"
            result += """<tr>
                        <td><b>{}</b></td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:red">{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:red">{}</td>
                    </tr>""".format(prediction.loc[i, "algorithm_name"], prediction.loc[i, "r_square"], prediction.loc[i, "mse"], prediction.loc[i, "rmse"], prediction.loc[i, "Fastest_shipment"], prediction.loc[i, "Avg_shipment"], Fastest_shipment_date.strftime("%b %d, %Y"), Avg_shipment_date.strftime("%b %d, %Y"), risk + " (" +prediction.loc[i, "accuracy_score"]+ ")")
        else:
            result += """<tr>
                        <td><b>{}</b></td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:green">{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:green">{}</td>
                    </tr>""".format(prediction.loc[i, "algorithm_name"], prediction.loc[i, "r_square"], prediction.loc[i, "mse"], prediction.loc[i, "rmse"], prediction.loc[i, "Fastest_shipment"], prediction.loc[i, "Avg_shipment"], Fastest_shipment_date.strftime("%b %d, %Y"), Avg_shipment_date.strftime("%b %d, %Y"), risk + " (" +prediction.loc[i, "accuracy_score"]+ ")")

    result += "</tbody></table>"
    return result

def print_result_with_ID3_algorithm(prediction, Delivery_Date):
    result = """<div style="margin-bottom:20px;  text-align: center;"><b>Prediction result</b></div>
                                        <table class="table table-bordered table-head-bg-info table-bordered-bd-info">
                                            <thead>
                                                <tr>
                                                    <th scope="col">Fastest shipment</th>
                                                    <th scope="col">Scheduled shipment</th>
                                                    <th scope="col">Fastest received goods date</th>
                                                    <th scope="col">Scheduled received goods date</th>
                                                    <th scope="col" style="WIDTH: 170px">Late Delivery Risk</th>
                                                </tr>
                                            </thead>
                                            <tbody>"""
    for i in range(1):
        Fastest_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Fastest_shipment"]))
        Avg_shipment_date = Delivery_Date + timedelta(days=int(prediction.loc[i, "Avg_shipment"]))
        risk = "NO"
        if(int(prediction.loc[i, "risk"]) == 1):
            risk = "YES"
            result += """<tr>
                        <td style="color:red">{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:red">{}</td>
                    </tr>""".format(prediction.loc[i, "Fastest_shipment"], prediction.loc[i, "Avg_shipment"], Fastest_shipment_date.strftime("%b %d, %Y"), Avg_shipment_date.strftime("%b %d, %Y"), risk + " (" +prediction.loc[i, "accuracy_score"]+ ")")
        else:
            result += """<tr>
                        <td style="color:green">{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td style="color:green">{}</td>
                    </tr>""".format(prediction.loc[i, "rmse"], prediction.loc[i, "Fastest_shipment"], prediction.loc[i, "Avg_shipment"], Fastest_shipment_date.strftime("%b %d, %Y"), Avg_shipment_date.strftime("%b %d, %Y"), risk + " (" +prediction.loc[i, "accuracy_score"]+ ")")

    result += "</tbody></table>"
    return result


class HomeView(View):
    def get(seft, request):
        return render(request, 'homepage/forms.html')


 
