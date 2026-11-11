# Test Queries Benchmark — Classical IR Retrieval Results

- **Queries Evaluated**: 34 across 5 Query Types (Natural Language, Short Keyword, Boolean, Exact Terminology, Spelling Errors)
- **Index Documents**: 77,565
- **Index Vocabulary**: 1,824,935 unique stemmed terms
- **BM25 Execution Time**: 25.92s (762.4ms/q)
- **BM25F Execution Time**: 46.89s (1379.0ms/q)
- **Multi-Level Execution Time**: 54.64s (1606.9ms/q)

## Performance By Query Type & Category

| Query ID | Type | Target Tech | Query Text | BM25 Top Score | BM25F Top Score | Top Retrieved Tech (BM25F) | Top Match Zone/Title |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NL_01** | `Natural_Language` | PyTorch | How to understand and debug CUDA memory... | 52.52 | **22.15** | PyTorch | Understanding CUDA Memory Usage ... |
| **NL_02** | `Natural_Language` | Kubernetes | Performing a rolling update on a Daemon... | 73.57 | **42.50** | Kubernetes | Perform a Rolling Update on a Da... |
| **NL_03** | `Natural_Language` | PostgreSQL | How to rebuild corrupted indexes concur... | 57.70 | **32.71** | PostgreSQL | PostgreSQL: Documentation: 12: R... |
| **NL_04** | `Natural_Language` | Python | Creating isolated Python virtual enviro... | 37.59 | **16.88** | Oracle Cloud | Manual and Offline Installations |
| **NL_05** | `Natural_Language` | FastAPI | Running asynchronous background notific... | 71.16 | **40.70** | FastAPI | Background Tasks - FastAPI |
| **KW_01** | `Keyword` | Docker | docker compose networking custom networ... | 34.62 | **20.44** | Docker | Networking in Compose | Docker Docs |
| **KW_02** | `Keyword` | RabbitMQ | rabbitmq consumer acknowledgements publ... | 65.89 | **37.56** | RabbitMQ | Consumer Acknowledgements and Pu... |
| **KW_03** | `Keyword` | Redis | redis job queue rust redis-rs Connectio... | 71.30 | **44.68** | Redis | Redis job queue with redis-rs | ... |
| **KW_04** | `Keyword` | Prometheus | prometheus monitoring docker container ... | 63.26 | **36.56** | Prometheus | Monitoring Docker container metr... |
| **KW_05** | `Keyword` | Rust | rust std sync Arc thread-safe reference... | 66.89 | **39.46** | Rust | Arc in std::sync - Rust |
| **BOOL_01** | `Boolean` | MongoDB | mongodb kubernetes operator TLS certifi... | 49.76 | **29.00** | MongoDB | Set Up a cert-manager Integratio... |
| **BOOL_02** | `Boolean` | Istio | istio ingress gateway AND passthrough N... | 37.41 | **21.54** | Istio | Istio / Ingress Gateway without ... |
| **BOOL_03** | `Boolean` | Go | golang database sql AND Executing SQL s... | 27.79 | **16.94** | Go | Documentation - The Go Programmi... |
| **BOOL_04** | `Boolean` | Terraform | terraform backend state locking AND s3 ... | 69.79 | **39.38** | Terraform | Upgrading to Terraform v1.10 | T... |
| **BOOL_05** | `Boolean` | Docker | docker host network driver AND networki... | 30.53 | **17.47** | Docker | Host network driver | Docker Docs |
| **EXACT_01** | `Exact_Terminology` | PostgreSQL | PostgreSQL pg_mcv_list_items Statistics... | 81.55 | **46.66** | PostgreSQL | PostgreSQL: Documentation: 18: 9... |
| **EXACT_02** | `Exact_Terminology` | Docker | Docker multi-stage builds Dockerfile CO... | 63.05 | **36.82** | Docker | Multi-stage builds | Docker Docs |
| **EXACT_03** | `Exact_Terminology` | PyTorch | PyTorch torch.distributed.fsdp.fully_sh... | 82.54 | **37.00** | PyTorch | torch.distributed.fsdp.fully_sha... |
| **EXACT_04** | `Exact_Terminology` | Linux Kernel | Linux Kernel /proc/sys/vm/drop_caches s... | 76.23 | **31.57** | Linux Kernel | Documentation for /proc/sys/vm/ ... |
| **EXACT_05** | `Exact_Terminology` | FastAPI | FastAPI dependency injection Depends() ... | 42.64 | **24.45** | FastAPI | Dependencies - FastAPI |
| **TYPO_01** | `Spelling_Error` | Kubernetes | kubernets taints and tolerations NoSche... | 60.79 | **34.43** | Kubernetes | Taints and Tolerations | Kubernetes |
| **TYPO_02** | `Spelling_Error` | PostgreSQL | postgresql jsonb contanment operater GI... | 37.61 | **21.88** | PostgreSQL | PostgreSQL: Documentation: 18: 8... |
| **TYPO_03** | `Spelling_Error` | PyTorch | pytorh torch.compile troubleshoting gra... | 49.11 | **21.77** | Hugging Face | torch.compile · Hugging Face |
| **TYPO_04** | `Spelling_Error` | Docker | dockr multi-stage bulds Dockerfile opti... | 49.67 | **28.97** | Docker | Multi-stage builds | Docker Docs |
| **TYPO_05** | `Spelling_Error` | RabbitMQ | rabitmq cluster formation and peer disc... | 37.55 | **22.02** | RabbitMQ | Cluster Formation and Peer Disco... |
| **SHORT_KW_01** | `Short_Keyword` | Docker | docker compose | 14.82 | **8.43** | Docker | Install the Docker Compose stand... |
| **SHORT_KW_02** | `Short_Keyword` | AWS | aws glue json | 21.62 | **12.45** | AWS | Using the JSON format in AWS Glu... |
| **SHORT_KW_03** | `Short_Keyword` | PyTorch | pytorch autograd function | 21.80 | **8.95** | PyTorch | Extending torch.func with autogr... |
| **SHORT_KW_09** | `Short_Keyword` | gRPC | grpc alts authentication | 23.38 | **13.38** | gRPC | ALTS authentication | Java | gRPC |
| **SHORT_KW_10** | `Short_Keyword` | React | react useEffect | 28.18 | **11.49** | OpenTelemetry | React Native App | OpenTelemetry |
| **SHORT_TYPO_01** | `Short_Spelling_Error` | Docker | dockr compse | 14.82 | **8.43** | Docker | Install the Docker Compose stand... |
| **SHORT_TYPO_02** | `Short_Spelling_Error` | AWS | aws opensearh | 13.40 | **7.58** | OpenSearch | JavaScript client | OpenSearch D... |
| **SHORT_TYPO_04** | `Short_Spelling_Error` | PyTorch | pytorh cuda memory | 18.90 | **9.23** | Hugging Face | Trainer · Hugging Face |
| **SHORT_TYPO_05** | `Short_Spelling_Error` | PostgreSQL | postgresql reindx | 16.99 | **9.70** | PostgreSQL | PostgreSQL: Documentation: 8.2: ... |
