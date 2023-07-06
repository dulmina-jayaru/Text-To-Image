from flask import Flask
from googletrans import Translator

def create_app():
    app = Flask(__name__)

    # Read API keys from apis.txt file
    def read_api_keys():
        api_keys = []
        with open("apis.txt", "r") as file:
            for line in file:
                api_keys.append(line.strip())
        return api_keys

    # Check if the provided API key is valid
    def is_valid_api_key(api_key):
        api_keys = read_api_keys()
        return api_key in api_keys

    # API endpoint for processing requests
    @app.route("/translate/<api_key>/<text>/<target_language>")
    def translate_text(api_key, text, target_language):
        # Check if the API key is valid
        if not is_valid_api_key(api_key):
            return "Invalid API key"

        # Perform the translation
        translator = Translator()
        translation = translator.translate(text, dest=target_language).text

        return translation

    return app

app = create_app()

if __name__ == "__main__":
    app.run()
