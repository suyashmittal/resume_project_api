import requests

# Define the URL of your Flask API
url = 'http://127.0.0.1:5000/predict/'

# Define the input data as a dictionary
data = {
    "categories": "python data science ai ml transformers natural language processing numpy seaborn matplotlib tensorflow keras html css javascript react node express mern"
}

# Send a POST request to the API with the input data
response = requests.post(url, json=data)

# Check the HTTP response status code
if response.status_code == 200:
    # Parse and print the JSON response (assuming it contains the prediction)
    prediction = response.json()
    print(prediction)
else:
    # Handle the case where the API request failed
    print(f'API Request Failed with Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')