# Simple Rule-Based Chatbot

def chatbot_response(user_input):
    user_input = user_input.lower()

    # Greetings
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"

    elif "how are you" in user_input:
        return "I'm just a chatbot, but I'm doing great! How about you?"

    # Asking about chatbot
    elif "who are you" in user_input:
        return "I am a simple rule-based chatbot created to answer your queries."

    # Farewell
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a nice day."

    # Default response
    else:
        return "Sorry, I don't understand that. Can you please rephrase?"

# Chat loop
print("Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    response = chatbot_response(user_input)
    print("Chatbot:", response)
    if "bye" in user_input.lower():
        break
