# Common Lambda Layer

This package provides common utilities for AWS Lambda functions, designed to be deployed as a Lambda Layer.

## Features

- **Logging**: Structured logging with request ID tracking
- **Parameters**: Utilities for retrieving parameters from SSM Parameter Store
- **Database**: PostgreSQL connection management
- **Decorators**: Utility decorators like execution time measurement

## Usage

### 1. Import in your Lambda function

```python
# Import specific utilities
from common.logging import get_custom_logger, set_request_id
from common.parameters import get_parameter
from common.database import get_connection
from common.decorators import measure_execution_time

# Or import everything
import common
```

### 2. Initialize logging with request ID

```python
logger = common.get_custom_logger("my.lambda.handler")

def lambda_handler(event, context):
    # Extract request ID from the event or context
    request_id = context.aws_request_id
    common.set_request_id(request_id)
    
    logger.info("Processing request")
    # ...
```

### 3. Measure function execution time

```python
@common.measure_execution_time()
def process_data(data):
    # Processing logic here
    return result
```

### 4. Access database

```python
def query_data():
    conn = common.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM my_table")
    result = cursor.fetchall()
    cursor.close()
    return result
```

## Deployment

To include this layer in your Lambda function:

1. Build the layer using the provided script:
   ```
   ./build_lambda_common_layer.sh
   ```

2. Deploy the layer to your AWS account.

3. Attach the layer to your Lambda function. 