FROM public.ecr.aws/lambda/python:3.10

RUN pip install tflite-runtime

ADD ["models", "./models"]
ADD ["scripts", "./scripts"]

CMD [ "./scripts", "lambda_function.lambda_handler" ]
