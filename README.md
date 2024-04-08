## About Sentiment Analysis
---
Sentiment Analysis is an API made using Flasgger. It predicts sentiment of inputted data.

It has four endpoints.
- **inputRNN**: user can input the text manually here, the system will predict using RNN
- **inputLSTM**: user can input the text manually here, the system will predict using LSTM
- **uploadCSV_RNN**: user can upload CSV file here, the system will predict using RNN
- **uploadCSV_LSTM**: user can upload CSV file here, the system will predict using LSTM

## <b>How to Run Sentiment Analysis</b>
---
1. Clone this repository, open command prompt
2. Create a virtual environment with python version of 3.10.9
> pip install virtualenv
> virtualenv -p python3.10.9 myenv
3. Enter the virtual environment
> myenv\Scripts\activate
4. Install module needed by running this code
> pip install -r requirements.txt --no-deps
5. Go to path below
> myenv\Lib\site-packages\keras\src\saving\legacy\saved_model\load_context.py
6. Rewrite "register_load_context_function" to "register_call_context_function" (line 68)
7. Run the app
> python sentiment_analysis.py
6. Open browser and go to link below
> http://127.0.0.1:5000/docs/

Successful deployment will result below.
<img src="img_md/app.png" alt="alt text" width="whatever" height="whatever"> 

## <center><b>Directory and File List</b></center>
---
<center><img src="img_md/ls.png" alt="alt text" width="whatever" height="whatever"></center>

## <center><b>Analysis Report</b></center>
---
