from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def HomeView(request):
    """ 
    Reading training data set. 
    """
    df = pd.read_csv('static/thyroid_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

    """ 
    Reading data from user. 
    """
    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        """ 
        Creating our training model. 
        """
        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = '1'
        elif int(predictions[0]) == 0:
            value = "0"

    return render(request,
                  'main/Thyroid.html',
                  {
                      'context': value,
                      'title': 'Breast Cancer Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'breast': True,
                      'background': 'bg-primary text-white'
                  })





