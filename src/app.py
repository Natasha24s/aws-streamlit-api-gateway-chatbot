import streamlit as st
import requests
import json

# Function to call the API
def call_api(query):
    api_url = "https://jqoatdawvh.execute-api.us-east-2.amazonaws.com/prod/chat"
    
    try:
        response = requests.post(api_url, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the API: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {str(e)}")
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit app
def main():
    st.set_page_config(page_title="Product AI Assistant", page_icon="🧠", layout="wide")

    st.title("🤖 Product AI Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Call API
            result = call_api(prompt)
            if result:
                full_response = result.get('generated_response', 'No response body available')
                message_placeholder.markdown(full_response)
            else:
                message_placeholder.error("Failed to get a valid response from the API.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display additional information
        with st.expander("See API call details"):
            st.json({
                "Query": result.get('query', 'N/A'),
                "Status Code": result.get('statusCode', 'N/A'),
                "Source": result.get('s3_location', 'N/A')
            })

if __name__ == "__main__":
    main()
