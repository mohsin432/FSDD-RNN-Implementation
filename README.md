# FSDD-RNN-Implementation


Usage python main.py [TaskA | TaskB | TaskC]


Folder Strtucture: 

project folder/
├── README.md # Project documentation 
├── requirements.txt # Dependencies of the project 
├── main.py # Python Main excutable  
├── train.py # train model script  
├── test.py # test model script  
├── dataprocessing/ # Dataset storage
  ├── DataDownloadandSplit.py # Data download and split into train test and validate parts
  └── SpokenMNISTData.py # Pre-processing of data
├── dataset/ # Data stored after preprocessing and train, test and validate split 
  ├── X_train.npy # Training features
  ├── y_train.npy # Training labels
  ├── X_val.npy # Validation features
  ├── y_val.npy # Validation labels
  ├── X_test.npy # Test features
  └── y_test.npy # Test labels
├── models/ # RNN Model definitions
  ├── Power2LSTM.py # model for Task3
  ├── QuantizeLSTM.py # model for Task2
  └── rnn_models.py # All RNN model implementations
├── output/ # Folder to store output models, some models are placed so tasks can run independently. 
├── utils/ # Utility functions
  ├── DisplayWeights.py # check model weights
  ├── Evaluate.py # run inference 
  └── MemoryCalc.py # calculate model memory consumption
