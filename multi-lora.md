# Deploy multi-LoRA adapters on LLMs

[View Markdown](/llm/serving/multi-lora.md)

# Deploy multi-LoRA adapters on LLMs

Learn how to deploy multiple [LoRA adapters](https://arxiv.org/abs/2106.09685) on a shared base model using Ray Serve LLM.

## Understand multi-LoRA deployment[​](#understand-multi-lora-deployment "Direct link to Understand multi-LoRA deployment")

Multi-LoRA lets your model switch between different fine-tuned adapters at runtime without reloading the base model.

Use multi-LoRA when your application needs to support multiple domains, users, or tasks using a single shared model backend. Following are the main reasons you might want to add adapters to your workflow:

* **Parameter efficiency**: LoRA adapters are small, typically less than 1% of the base model's size. This makes them cheap to store, quick to load, and easy to swap in and out during inference, which is especially useful when memory is tight.

* **Runtime adaptation**: With multi-LoRA, you can switch between different adapters at inference time without reloading the base model. This allows for dynamic behavior depending on user, task, domain, or context, all from a single deployment.

* **Simpler MLOps**: Multi-LoRA cuts down on infrastructure complexity and cost by centralizing inference around one model.

## Prepare LoRA adapter checkpoints[​](#prepare-lora-adapter-checkpoints "Direct link to Prepare LoRA adapter checkpoints")

Ray Serve LLM can load your LoRA adapters at runtime directly from cloud storage.

note

Ray Serve LLM supports AWS S3, Google Cloud Storage, and Azure Blob or data lake storage.

Make sure each adapter checkpoint contains at least the following files:

* **`adapter_config.json`**: Specifies the LoRA architecture, including rank, target modules, scaling, and bias configuration.

* **`adapter_model.safetensors`**: Contains the trained LoRA weights to inject into the base model during inference.

If either file is missing, the backend vLLM engine can't load the adapter.

Your cloud storage should look like this:

```
[s3|gs|abfss|azure]://my/dynamic/lora/loading/path/
├── adapter_name_1/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── adapter_name_2/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── ...
```

### Find LoRA adapters on Hugging Face[​](#find-lora-adapters-on-hugging-face "Direct link to Find LoRA adapters on Hugging Face")

If you pick a base model hosted on Hugging Face, you can view publicly available adapters directly on the model's page. You can then download the checkpoint with `huggingface_hub.snapshot_download()`.

```
from huggingface_hub import snapshot_download
adapters = {
    "adapter_name_1": "huggingface/adapter_id_1",
    "adapter_name_2": "huggingface/adapter_id_2",
    "adapter_name_3": "huggingface/adapter_id_3"
}

for name, repo in adapters.items():
    local_path = snapshot_download(repo)
    # Upload local_path to your cloud storage
    ...
```

## Configure Ray Serve LLM with multi-LoRA[​](#configure-ray-serve-llm-with-multi-lora "Direct link to Configure Ray Serve LLM with multi-LoRA")

To enable multi-LoRA on your deployment, update your Ray Serve LLM configuration with a few extra settings. See [Ray Serve LLM multi-LoRA guidelines](https://docs.ray.io/en/latest/serve/llm/user-guides/multi-lora.html) for more details.

### LoRA configuration[​](#lora-configuration "Direct link to LoRA configuration")

* Specify `dynamic_lora_loading_path` to your AWS S3, Google Cloud, or Azure storage URI.
* (Optional) Set `max_num_adapters_per_replica` to limit the number of adapters loaded per replica. This must match `max_loras` you set in your engine configuration.

note

Make sure to add the right prefix to your `dynamic_lora_loading_path` URI. This must be one of `s3://` (AWS), `gs://` (Google Cloud), `abfss://` (Azure DFS), or `azure://` (Azure DFS or Blob Storage)

### Runtime environment[​](#runtime-environment "Direct link to Runtime environment")

Add any cloud-specific environment variables required to access your storage, such as credentials, regions, or account identifiers. These depend on your cloud provider and authentication method.

### Engine arguments[​](#engine-arguments "Direct link to Engine arguments")

Forward these parameters to your vLLM engine:

* Set `enable_lora: true` in your `engine_kwargs`.
* Set `max_lora_rank` to the highest LoRA rank you plan to use.
* (Optional) Set `max_loras` to limit the number of adapters loaded per replica. This must match `max_num_adapters_per_replica` you set in your LoRA configuration.

See the [Ray Serve LLM](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LoraConfig.html#ray-serve-llm-loraconfig) and [vLLM](https://docs.vllm.ai/en/stable/configuration/engine_args.html#loraconfig) APIs for the full list of supported parameters.

* Ray Serve LLM Python SDK
* Anyscale Service config file

```
from ray.serve.llm import LLMConfig

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-base-model-id",
        ...
    ),
    ...
    lora_config=dict(
        dynamic_lora_loading_path=<YOUR-S3/GS/AZURE-URI>, #s3://, gs://, abfss:// or azure://
        max_num_adapters_per_replica=32  # (Optional)
    ),
    runtime_env=dict(
        env_vars={
          ...
          "<CLOUD-SPECIFIC-VARIABLE>": ...
          # "AWS_REGION": <YOUR-AWS-S3-REGION> # if using S3
          # "AZURE_TENANT_ID": <YOUR-AZURE-TENANT-ID> # if using Azure
          # ...
        }
    ),
    engine_kwargs=dict(
        ...
        enable_lora=True,
        max_lora_rank=32, # Set to the largest rank you plan to use.
        max_loras=32  # Need to set this to the same value as `max_num_adapters_per_replica`.
    ),
)
```

```
# service.yaml
## Anyscale Cloud config
name: deploy-multi-lora-service
image_uri: anyscale/ray-llm:2.49.0-py311-cu128   # Or use `containerfile` pointing to your custom Dockerfile
compute_config:
  ... # Your compute config
working_dir: .
cloud: # Your default cloud if empty

## Ray Serve LLM config
applications:
- ...
  args:
    llm_configs:
      - model_loading_config:
          model_id: my-base-model-id
        ...
        lora_config:
          dynamic_lora_loading_path: <YOUR-S3/GS/AZURE-URI> #s3://, gs://, abfss:// or azure://
          max_num_adapters_per_replica: 32 # (Optional)
        runtime_env:
          env_vars:
            ...
            <CLOUD-SPECIFIC-VARIABLE>: ...
            # AWS_REGION: <YOUR-AWS-S3-REGION> # if using S3
            # AZURE_TENANT_ID: <YOUR-AZURE-TENANT-ID> # if using Azure
            # ...
        engine_kwargs:
          ...
          enable_lora: True
          max_loras: 32 # Need to set this to the same value as `max_num_adapters_per_replica`.
```

## Send requests to multi-LoRA adapters[​](#send-requests-to-multi-lora-adapters "Direct link to Send requests to multi-LoRA adapters")

You can deploy your model as usual. For guidance depending on your model size, see [Examples of Ray Serve LLM applications](https://docs.ray.io/en/latest/serve/examples.html).

To query the base model, call your service as you normally would.

To use a specific LoRA adapter at inference time, include the adapter name in your request using the format:

```
<base_model_id>:<adapter_name>
```

Here:

* `<base_model_id>` is the `model_id` defined in your Ray Serve LLM configuration.
* `<adapter_name>` is the adapter's folder name in your cloud storage.

### Example queries[​](#example-queries "Direct link to Example queries")

Query both the base model and different LoRA adapters:

```
from openai import OpenAI

client = OpenAI(...)

# Base model request (no adapter)
response = client.chat.completions.create(
    model="my-base-model-id",  # no adapter
    ...
)

# Adapter 1
response = client.chat.completions.create(
    model="my-base-model-id:adapter_name_1", # Follow naming convention in your cloud storage
    ...
)

# Adapter 2
response = client.chat.completions.create(
    model="my-base-model-id:adapter_name_2", # Follow naming convention in your cloud storage
    ...
)
```

## Apply best practices[​](#apply-best-practices "Direct link to Apply best practices")

Follow these suggestions to get more reliable and efficient behavior when using multi-LoRA in Ray Serve LLM:

### Implement adapter routing strategies[​](#implement-adapter-routing-strategies "Direct link to Implement adapter routing strategies")

Implement adapter selection logic to route the request to the appropriate LoRA adapter based on the task, user, or NLP logic.

```
adapter_name = adapter_router(messages, **kwargs)
response = client.chat.completions.create(
    model=f"my-base-model-id:{adapter_name}",
    messages=messages,
    ...
)
```

### Batch requests by adapter[​](#batch-requests-by-adapter "Direct link to Batch requests by adapter")

To maximize throughput, group requests by adapter name and batch them together.

### Use a low temperature for structured outputs[​](#use-a-low-temperature-for-structured-outputs "Direct link to Use a low temperature for structured outputs")

Consider decreasing the `temperature` of your model when you expect structured outputs. This can help create consistent behavior across adapters.

### Limit the number of requested adapters per deployment[​](#limit-the-number-of-requested-adapters-per-deployment "Direct link to Limit the number of requested adapters per deployment")

To avoid memory overload and maintain predictable latency when using multiple adapters, set reasonable limits on:

* How many adapters each replica can load, with `max_num_adapters_per_replica` (Ray Serve LLM parameter).
* How many adapters the engine can request per batch, with `max_loras` (vLLM engine parameter).

These two limits must match for correct behavior:

```
applications:
- ...
  args:
    llm_configs:
        ...
        lora_config:
          ...
          ## Maximum number of LoRA loaded in memory per replica
          max_num_adapters_per_replica: 16 
        engine_kwargs:
          ...
          enable_lora: True
          ## Maximum number of LoRA loaded per batch of request
          max_loras: 16 # must match `max_num_adapters_per_replica`
```

Adjust the limits based on your GPU memory usage or how many adapters your users typically need.

### Warm up adapters[​](#warm-up-adapters "Direct link to Warm up adapters")

vLLM uses an LRU cache to keep up to `max_cpu_loras` LoRA adapters in memory. To reduce cold-start latency, pre-load adapters by sending a dummy request with one token. This helps keep them fresh in the cache.

If needed, you can also increase `max_cpu_loras` in your config to allow more adapters to stay in memory. Make sure `max_cpu_loras` is greater than `max_loras` and `max_num_adapters_per_replica` defined in the preceding sections.

```
applications:
- ...
  args:
    llm_configs:
        ...
        lora_config:
          ...
          ## Maximum number of LoRA per batch sent to a replica
          max_num_adapters_per_replica: 16 
        engine_kwargs:
          ...
          enable_lora: True
          ## Maximum number of LoRA per batch of request
          max_loras: 16 # must be less than `max_cpu_loras` and match `max_num_adapters_per_replica`
          ## Maximum number of LoRA kept in memory
          max_cpu_loras: 32 # must be more than `max_loras`
```

### Advanced usage[​](#advanced-usage "Direct link to Advanced usage")

For more control over your LoRA deployment, see the [Ray Serve LLM](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LoraConfig.html#ray-serve-llm-loraconfig) and [vLLM](https://docs.vllm.ai/en/latest/configuration/engine_args.html#loraconfig) documentation for the full list of supported LoRA configuration parameters. You can configure your vLLM engine in the `engine_kwargs` section of your Ray Serve configuration.

## Summary[​](#summary "Direct link to Summary")

In this guide, you learned how to deploy a shared base model with multiple LoRA adapters using Ray Serve LLM. You learned how to set up your cloud-hosted checkpoints, route requests to different adapters, and apply best practices to improve your multi-LoRA workflow.
