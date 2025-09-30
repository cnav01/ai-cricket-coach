import google.generativeai as genai

# Make sure you've configured the API key already
genai.configure(api_key="AIzaSyBsjzciUdls9zxzQ2v3FpDmtkVhLGAdQIw")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print("Valid model:", m.name, m.supported_generation_methods)
