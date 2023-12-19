# Create Docker Container

Build docker image (using `Dockerfile`), tag it as `latest` and run it.

```bash
(base) $~> docker build --platform linux/amd64 -t facial-emotion-recognition .
(base) $~> docker run -it --rm -p 8080:8080 --platform linux/amd64 facial-emotion-recognition:latest
```

Use `docker images` to confirm this container image has been built.
```
(base) $~> docker images
REPOSITORY                                                                 TAG                              IMAGE ID       CREATED         SIZE
facial-emotion-recognition                                                 latest                           0c7d339b16ae   44 hours ago    751MB
docker/getting-started                                                     latest                           157095baba98   20 months ago   27.4MB
```

# Create AWS ECR using command line

[AWS ECR: Console](https://eu-north-1.console.aws.amazon.com/ecr/get-started)

Install AWS CLI utility: 
```bash
(base) $~> pip install awscli
```

Configure, if running for the first time.
* Create security credentials here: [IAM > Security credentials](https://us-east-1.console.aws.amazon.com/iam/home#/security_credentials?section=IAM_credentials).
```bash
(base) $~> aws configure
AWS Access Key ID [None]: AKIA****
AWS Secret Access Key [None]: ****
Default region name [None]: eu-north-1
Default output format [None]:
```

Create ECR [Elastic Container Registry]:

```bash
(base) $~> aws ecr create-repository --repository-name facial-emotion-recognition
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:eu-north-1:166783209982:repository/facial-emotion-recognition",
        "registryId": "166783209982",
        "repositoryName": "facial-emotion-recognition",
        "repositoryUri": "166783209982.dkr.ecr.eu-north-1.amazonaws.com/facial-emotion-recognition",
        "createdAt": 1703003010.0,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```
It generates the above response. The important part is: `"repositoryUri"`, which we use to construct the `${REMOTE_URI}` and will use later while publishing the docker container to the ECR created.

```
ACCOUNT=166783209982
REGION=eu-north-1
REGISTRY=facial-emotion-recognition
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=facial-emotion-recognition-001
REMOTE_URI=${PREFIX}:${TAG}
```

The registry, called **facial-emotion-recognition** should now be created, and should be reflected [here](https://eu-north-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-north-1). However, if you click the repository, there are no image tags within. [⚠️ Make sure region is same as one used for `aws configure`]. 

# Publish Docker Container Image to AWS ECR

Log-in to registry [`${PASSWORD}` should be replaced by the actual password]:

```bash
(base) $~> $(aws ecr get-login --no-include-email)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded
```

This indicates the log-in was succesful.

[OPTIONAL] If you just want to see the output of the above command, use:
```bash
(base) $~> aws ecr get-login --no-include-email
docker login -u AWS -p PASSWORD https://166783209982.dkr.ecr.eu-north-1.amazonaws.com
```

Execute the lines saved from `"repositoryUri"` on the command line to get the `${REMOTE_URI}`. An `echo` command should give back the entire URI. 

```
(base) $~> echo ${REMOTE_URI}
166783209982.dkr.ecr.eu-north-1.amazonaws.com/facial-emotion-recognition:facial-emotion-recognition-001
```

Tag the docker image (with tag `latest`) created earlier with the ${REMOTE_URI} created above.

```bash
(base) $~> docker tag facial-emotion-recognition:latest ${REMOTE_URI}
```

and push to ${REMOTE_URI} [⚠️ Takes 5 minutes]

```bash
(base) $~> docker push ${REMOTE_URI}
The push refers to repository [166783209982.dkr.ecr.eu-north-1.amazonaws.com/facial-emotion-recognition]
41f7da4301db: Pushed 
.
.
. 
5198865f908d: Pushed 
facial-emotion-recognition-001: digest: sha256:ac43b36001da3832a776f900b919c87b969d5f8921baf5dfc11105c0b147c5a0 size: 2423
```

On the [ECR console](https://eu-north-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-north-1), under the repository **facial-emotion-recognition**, there should now be one image tag: **facial-emotion-recognition-001**.

# Create AWS Lambda function

To create, configure and test the Lambda function, we follow the instructions in this [video](https://youtu.be/kBch5oD5BkY?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&t=457). We test it out on the above url, and get identical results.

# Create and configure AWS API Gateway

We follow the instructions [here](https://youtu.be/wyZ9aqQOXvs?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

* to create a REST API (POST METHOD: /predict ) with the Lambda function created above
* test it using the above url and get the same predictions back
* deploy it as 'new stage' test which gives the invoke url: https://xowsdry1bc.execute-api.eu-north-1.amazonaws.com/Test
* test the API gateway using the `scripts/test-aws-rest-api.py` script.

```bash
(base) $~> python test-aws-rest-api.py 
{'Anger': 8.251844068965241e-35, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 1.0, 'Neutral': 0.0, 'Sadness': 0.0, 'Surprise': 0.0}
```
