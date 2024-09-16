# California Housing Prices Machine Learning Model
 
I created a tensorflow machine learning neural network trained on a [kaggle dataset] (https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset?select=realtor-data.zip.csv) that provided data collected from (https://www.realtor.com/). The dataset consisted of the following categories:
- brokered by (categorically encoded agency/broker)
- status (Housing status - a. ready for sale or b. ready to build)
- price (Housing price, it is either the current listing price or recently sold price if the house is sold recently)
- bed (# of beds)
- bath (# of bathrooms)
- acre_lot (Property / Land size in acres)
- street (categorically encoded street address)
- city (city name)
- state (state name)
- zip_code (postal code of the area)
- house_size (house area/size/living space in square feet)
- prev_sold_date (Previously sold date)

The model predicts the housing price based on the # of beds, # of bathrooms, acre_lot, house_size, zip_code and city, prev_sold_date. I specifically trained the machine learning model on California houses, as the dataset was over 2 million rows, which couldn't be entirely stored within a single CSV file.

## Cleaning the Data
Using Pandas, I deleted all rows that contained any null data. Then, I utilized the StringLookup() function from Tensorflow to convert the city name attribute from a string into a unique sequence of numbers. I also decomposed the previous sold date into just the previous sold year since the day and month a house sold shouldn't hold much weight. Afterward, I utilized the sklearn package to apply a StandardScaler to my data, allowing larger values such as zip_code and city name to be weighted like acre_lot and other smaller values within my data algorithm. 

## Training the Model
For data distribution, I employed an 80-20 split, allocating 80% of the data to the training set and dividing the remaining 20% between the validation and testing sets. This distribution was chosen to ensure a robust model that is trained on a significant portion of the data, while also being validated and tested on a representative sample. 
For the neural network, I utilized a 5-layer neural network with relu activation function for each hidden layer. I decided to use 300 epochs that stop training after there are ten instances where the model doesn't improve its loss. 

## Results of the Model
After evaluating the ML model with my test data, the average test loss (mean squared error) was around 160,000, and the model had an R-squared value of 0.69. Hopefully, as I learn more about TensorFlow, I can better optimize my function to lower the test loss and improve the R-squared without overfitting the model. 

## Front-end
After I curated my machine learning model, I developed a front-end using Flask that uses my TensorFlow model to predict California Housing Prices based on data inputted. 
