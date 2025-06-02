def serve_model(model_path):
    import torch
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # Load the quantized model
    model = torch.load(model_path)
    model.eval()

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        input_text = data.get('input_text', '')

        # Perform inference
        with torch.no_grad():
            output = model(input_text)

        return jsonify({'output': output})

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)