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

The model predicts the housing price based on the # of beds, # of bathrooms, acre_lot, house_size, zip_code and city, prev_sold_date. I chose to specifically train the machine learning model on just California houses, as the entire dataset was over 2 million rows, which couldn't be fully stored within a single csv file. 

## Cleaning the Data
Using Pandas, I deleted all rows that contained any null data, and then utilized the StringLookup() function from tensorflow to convert the city name attribute from a string into a unique sequence of numbers. I also decomposed the previous sold date into just the previous sold year, since the day and month that a house was sold shouldn't hold much weight. Afterwards, I utilized the sklearn package to apply a StandardScaler to my data, allowing larger values such as zip_code and city name to be weighted the same as acre_lot and other smaller values within my data algorithm. 

## Training the Model
I utilized a train, test, validation split, placing 80% of my data within the training dataset and splitting the remaining 20% into the validation set and testing set. 
For the neural network, I utilized a 5 layer neural network with relu activation function for each hidden layer. I decided to use 300 epochs that stops training after there are 10 instances where the model doesn't improve its loss. 

## Results of the Model
After evaluating the ML model with my test data, the average test loss (mean squared error) was around 160,000, and the model had an R-squared value of 0.67. Hopefully as I learn more about TensorFlow, I can better optimize my function to lower the test loss and improve the R-squared, without overfitting the model. 

## Front-end
After I curated my machine learning model, I developed a front-end using Flask that uses my TensorFlow model to predict California Housing Prices based on data inputted. 
