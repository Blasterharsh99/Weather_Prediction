from roboflow import Roboflow
import pdb
rf = Roboflow(api_key="hg2tn41W0JJi3MR9yK7q")
project = rf.workspace().project("weather-classification-fa0p4")
model = project.version(1).model

# infer on a local image
#pdb.set_trace()
y = model.predict(r'cloud_test.jpg').json()
x=y['predictions'][0].get('top')
# infer on an image hosted elsewhere
#x = model.predict("", hosted=True).json()

if x == 'Cloudy':
    print("High Rain Chance")
elif x == 'shinny':
    print("warm weather")
elif x == 'Rain':
    print("It's Raining")
else:
    print(x)

# save an image annotated with your predictions
#model.predict(r'cloud_test.jpg').save("prediction.jpg")