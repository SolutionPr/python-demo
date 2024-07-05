
import requests
import pandas as pd
import plotly.graph_objects as go
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
import re

# Initialize the OpenAI LLM (replace 'your-openai-api-key' with your actual API key)
llm = OpenAI(api_key="")

# Define a simpler prompt template
prompt_template = """
Using Apple stock price data, answer the question: {question}
"""

# Create a LangChain prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

# Create a LangChain LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

def fetch_stock_data(api_key, ticker="AAPL", start_date=None, end_date=None):
    """Fetch the stock data for the given ticker and date range."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            return pd.DataFrame(data['results'])
        else:
            print("No results found in the response.")
            return pd.DataFrame()
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()

def get_stock_data(df):
    """Get the stock data as a string."""
    return df.to_string(index=False)

def parse_period(period_str):
    """Parse the period string to calculate the start date."""
    period_map = {
        'day': 1,
        'month': 30,
        'year': 365
    }
    
    pattern = re.compile(r"(\d+)\s*(day|month|year)")
    matches = pattern.findall(period_str.lower())
    
    total_days = 0
    for match in matches:
        num, unit = match
        total_days += int(num) * period_map[unit]
    
    return total_days

# Function to handle user queries
def handle_query(api_key, question, ticker="AAPL"):
    period_pattern = re.compile(r"last\s*(.*)")
    match = period_pattern.search(question.lower())
    
    if match:
        period_str = match.group(1)
        total_days = parse_period(period_str)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)
        
        df = fetch_stock_data(api_key, ticker, start_date, end_date)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            stock_data = get_stock_data(df[['date', 'c']])
            inputs = {
                "question": question
            }
            response = chain.run(inputs)
            
            # Generate the plot using Plotly
            fig = go.Figure()
            
            # Create traces for up and down prices
            fig.add_trace(go.Scatter(x=df['date'], y=df['c'], mode='lines+markers', name='Apple Stock Price',
                                     line=dict(color='green', width=2) if df['c'].iloc[-1] >= df['c'].iloc[0] else dict(color='red', width=2)))
            
            fig.update_layout(title=f'Apple Stock Prices from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                              xaxis_title='Date',
                              yaxis_title='Closing Price (USD)',
                              showlegend=True,
                              template='plotly_dark')
            fig.show()
            
            return response
        else:
            return "Failed to fetch stock data."
    else:
        return "Invalid period format in the question."

# Example usage
api_key = "eEwCbZyN0_omcUAGMvr_fNdK4BC9PGyz"

# Prompt the user for input
question = input("Enter your question: ")

# Handle the query
response = handle_query(api_key, question)
print(response)