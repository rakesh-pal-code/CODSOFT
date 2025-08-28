import streamlit as st

# Function for chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"

    elif "how are you" in user_input:
        return "I'm just a chatbot, but I'm doing great! How about you?"

    elif "who are you" in user_input:
        return "I am a simple rule-based chatbot created to answer your queries."

    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a nice day."

    else:
        return "Sorry, I don't understand that. Can you please rephrase?"


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Rule-Based Chatbot", page_icon="🤖")

# Custom CSS for styling
st.markdown("""
    <style>
    .chat-container {
        max-width: 600px;
        margin: auto;
        padding: 20px;
        border-radius: 15px;
        background: #f9f9f9;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .user-msg {
        background: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-msg {
        background: #e9ecef;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Simple Rule-Based Chatbot")

# Chat history stored in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = chatbot_response(user_input)
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
