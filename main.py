import os
import io
import warnings
import base64
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from flask import Flask, jsonify , request
def create_app():
    app = Flask(__name__)
    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key="API_KEY",  
        verbose=True, 
        engine="stable-diffusion-xl-1024-v0-9",
    )
    
    
    @app.route('/generate_image', methods=['GET'])
    def generate_image():
        prompt = request.args.get('prompt', '')  
        # Set up our initial generation parameters.
        answers = stability_api.generate(
            prompt=prompt,
            seed=992446758,
            steps=50,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )
    
        image_data = None
    
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
        return jsonify({'image_data': image_data})

    return app

app = create_app()

if __name__ == "__main__":
    app.run()
