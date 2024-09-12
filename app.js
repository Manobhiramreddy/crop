const express = require('express');
const bodyParser = require('body-parser');
const { StandardScaler, MinMaxScaler } = require('scikit-learn');
const fs = require('fs');

const app = express();
const port = 3000;

const model = fs.readFileSync('model.pkl'); // Assuming the model is stored in a file
const sc = fs.readFileSync('standscaler.pkl'); // Assuming the scaler files are stored in a file
const mx = fs.readFileSync('minmaxscaler.pkl'); // Assuming the scaler files are stored in a file

app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
  res.sendFile(__dirname + "/index.html");
});

app.post('/predict', (req, res) => {
  const { Nitrogen, Phosporus, Potassium, Temperature, Humidity, pH, Rainfall } = req.body;

  const featureList = [Nitrogen, Phosporus, Potassium, Temperature, Humidity, pH, Rainfall];
  const singlePred = featureList.map(Number); // Ensure values are numbers

  // Perform scaling as per your Flask code
  const mxFeatures = mx.transform([singlePred]);
  const scMxFeatures = sc.transform(mxFeatures);

  // Use the model to predict
  const prediction = model.predict(scMxFeatures);

  const cropDict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
  };

  let result;
  if (cropDict.hasOwnProperty(prediction[0])) {
    const crop = cropDict[prediction[0]];
    result = `${crop} is the best crop to be cultivated.`;
  } else {
    result = "Sorry, we could not determine the best crop to be cultivated with the provided data.";
  }

  res.render('index.html', { result });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
