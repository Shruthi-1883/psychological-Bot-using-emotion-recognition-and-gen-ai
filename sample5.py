import speech_recognition as sr
import pyttsx3
import boto3
import json

# Initialize the recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Function to listen to the user input via the microphone
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network.")
        return ""

# Function to speak the generated response
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to generate a response using Bedrock
def generate_response(prompt):
    # Define the request payload with the model's native structure
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    try:
        # Convert the native request to JSON
        request = json.dumps(native_request)

        # Invoke the model and stream the response
        streaming_response = client.invoke_model_with_response_stream(
            modelId=model_id, body=request
        )

        # Collect the response in real-time
        full_response = ""
        for event in streaming_response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                full_response += chunk["delta"].get("text", "")
                print(chunk["delta"].get("text", ""), end="")  # Print chunk of text
        
        return full_response
    except Exception as e:
        print(f"Error invoking Claude model: {e}")
        return "Sorry, I encountered an error while generating a response."

# Main loop for listening and responding
if __name__ == "__main__":
    while True:
        command = listen()  # Listen to the user's input
        if command.lower() in ["exit", "quit"]:
            speak("Goodbye!")  # Exit the loop if the user says "exit" or "quit"
            break
        elif command:
            # Generate a response using Bedrock
            response = generate_response(command)
            print(f"Claude says: {response}")  # Print the response
            speak(response)  # Speak out the response

