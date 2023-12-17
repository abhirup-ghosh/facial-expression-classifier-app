# Use the official AWS Lambda Python 3.10 base image from Amazon Elastic Container Registry (ECR)
FROM public.ecr.aws/lambda/python:3.10

# Install the specified version of the Pillow library using pip
RUN pip install pillow==10.0.1

# Install TensorFlow Lite runtime from a specific GitHub repository release
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# Copy the TensorFlow Lite model file "emotion_classifier.tflite" to the root directory of the container
COPY models/emotion_classifier.tflite .

# Copy the Python script "lambda_function.py" from the "scripts" directory to the root directory of the container
COPY scripts/lambda_function.py .

# Set the default command to execute the Lambda handler function in "lambda_function.py"
CMD ["lambda_function.lambda_handler"]
