import streamlit as st
from api.aws_api_gateway import send_message
from utils.helpers import format_bot_message, format_user_message

def main():
    st.title("AWS API Gateway Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.write(message)

    # User input
    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append(format_user_message(user_input))

            # Send message to API and get response
            response = send_message(user_input)

            if response:
                bot_response = response.get("message", "Sorry, I didn't understand that.")
                # Add bot response to chat history
                st.session_state.messages.append(format_bot_message(bot_response))
            else:
                st.error("Failed to get response from the API.")

            # Clear user input
            st.session_state.user_input = ""

            # Rerun the app to update the chat display
            st.experimental_rerun()

if __name__ == "__main__":
    main()