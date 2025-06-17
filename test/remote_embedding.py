import boto3
import json

# Initialize the SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')

# Your endpoint name (from CloudFormation output)
endpoint_name = 'embedding-endpoint'

# Prepare your input data
payload = {
    "inputs": ["Your text to embed here"]

}

try:
    # Call the endpoint
    response = runtime.invoke_endpoint(
        EndpointName='embedding-endpoint',
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    # Parse the response
    result = json.loads(response['Body'].read().decode())

    # The embeddings will be in the result
    embeddings = result  # This will be a list of numbers (the embedding_provider vector)

    print(f"Embedding dimensions: {len(embeddings)}")
    print(f"First 5 values: {embeddings[:5]}")

except Exception as e:
    print(f"Error: {e}")