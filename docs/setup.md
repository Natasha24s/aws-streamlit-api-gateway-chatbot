# Setup Instructions

1. Clone the repository:

    

    
git clone https://github.com/yourusername/aws-streamlit-chatbot.git
cd aws-streamlit-chatbot

    

2. Create a virtual environment and activate it:

    

    
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

    

3. Install the required packages:

    

    
pip install -r requirements.txt

    

4. Set up your AWS API Gateway endpoint:
- Create a `.env` file in the root directory
- Add your API endpoint URL:
  ```
  API_ENDPOINT=https://your-api-gateway-endpoint.amazonaws.com/stage
  ```

5. Run the Streamlit app:

    

    
streamlit run src/app.py

    

The app should now be running on `http://localhost:8501`.

    

    
    
