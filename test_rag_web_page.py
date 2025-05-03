# Query Data
import ollama
response = ollama.chat(model='gemma3:4b', 
    messages=[{
        'role': 'user', 
        'content': 'Describe the image',
    }],
    # options={"temperature":0.7}
    )
print(response)
