# The decoder as a Lambda container image.
#
# A zip stopped fitting: OpenCV 5 and numpy are 211 MB unpacked, the CRNN text
# model is 34 MB, and Lambda's zip limit is 250 MB. An image may be 10 GB.
#
#   docker build -t arpi .                       # build
#   docker run -p 9000:8080 arpi                 # local Lambda emulator
#
# CDK builds this itself on deploy (infra/arpi_stack.py), so the commands
# above are only for testing the image by hand.
#
# Python 3.12 is the Lambda runtime, not the newest. Pins match
# requirements.txt.
FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir \
        opencv-python-headless==5.0.0.93 \
        numpy==2.5.2 \
        pyyaml==6.0.3

# Code, page, config and model. The model is fetched by
# tools/fetch_models.py before the build and checked against its hash there.
COPY src/arpi ${LAMBDA_TASK_ROOT}/arpi
COPY web ${LAMBDA_TASK_ROOT}/web
COPY infra/config.yaml ${LAMBDA_TASK_ROOT}/infra/config.yaml
COPY models/text_recognition_CRNN_EN_2021sep.onnx ${LAMBDA_TASK_ROOT}/models/

CMD ["arpi.handler.handler"]
