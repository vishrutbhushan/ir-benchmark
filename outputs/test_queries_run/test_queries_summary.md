# Test Queries Benchmark — Classical IR Retrieval Results

- **Queries Evaluated**: 34 across 5 Query Types (Natural Language, Short Keyword, Boolean, Exact Terminology, Spelling Errors)
- **Index Documents**: 61,966
- **Index Vocabulary**: 1,799,233 unique stemmed terms
- **BM25 Execution Time**: 6.10s (179.5ms/q)
- **BM25F Execution Time**: 7.88s (231.8ms/q)
- **Multi-Level Execution Time**: 10.13s (298.1ms/q)

## Performance By Query Type & Category

| Query ID | Type | Target Tech | Query Text | BM25 Top Score | BM25F Top Score | Top Retrieved Tech (BM25F) | Top Match Zone/Title |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NL_01** | `Natural_Language` | PyTorch | How to understand and debug CUDA memory... | 55.81 | **23.60** | PyTorch | Understanding CUDA Memory Usage ... |
| **NL_02** | `Natural_Language` | Kubernetes | Performing a rolling update on a Daemon... | 74.30 | **42.95** | Kubernetes | Perform a Rolling Update on a Da... |
| **NL_03** | `Natural_Language` | PostgreSQL | How to rebuild corrupted indexes concur... | 59.08 | **33.47** | PostgreSQL | PostgreSQL: Documentation: 12: R... |
| **NL_04** | `Natural_Language` | Python | Creating isolated Python virtual enviro... | 38.58 | **17.33** | Oracle Cloud | Manual and Offline Installations |
| **NL_05** | `Natural_Language` | FastAPI | Running asynchronous background notific... | 71.90 | **41.13** | FastAPI | Background Tasks - FastAPI |
| **KW_01** | `Keyword` | Docker | docker compose networking custom networ... | 34.91 | **20.65** | Docker | Networking in Compose | Docker Docs |
| **KW_02** | `Keyword` | RabbitMQ | rabbitmq consumer acknowledgements publ... | 66.57 | **37.92** | RabbitMQ | Consumer Acknowledgements and Pu... |
| **KW_03** | `Keyword` | Redis | redis job queue rust redis-rs Connectio... | 71.30 | **44.72** | Redis | Redis job queue with redis-rs | ... |
| **KW_04** | `Keyword` | Prometheus | prometheus monitoring docker container ... | 63.67 | **36.80** | Prometheus | Monitoring Docker container metr... |
| **KW_05** | `Keyword` | Rust | rust std sync Arc thread-safe reference... | 66.26 | **39.06** | Rust | Arc in std::sync - Rust |
| **BOOL_01** | `Boolean` | MongoDB | mongodb kubernetes operator TLS certifi... | 49.83 | **29.08** | MongoDB | Set Up a cert-manager Integratio... |
| **BOOL_02** | `Boolean` | Istio | istio ingress gateway AND passthrough N... | 37.20 | **21.53** | Istio | Istio / Ingress Gateway without ... |
| **BOOL_03** | `Boolean` | Go | golang database sql AND Executing SQL s... | 27.74 | **16.95** | Go | Documentation - The Go Programmi... |
| **BOOL_04** | `Boolean` | Terraform | terraform backend state locking AND s3 ... | 69.69 | **39.28** | Terraform | Upgrading to Terraform v1.10 | T... |
| **BOOL_05** | `Boolean` | Docker | docker host network driver AND networki... | 30.70 | **17.56** | Docker | Host network driver | Docker Docs |
| **EXACT_01** | `Exact_Terminology` | PostgreSQL | PostgreSQL pg_mcv_list_items Statistics... | 82.65 | **47.30** | PostgreSQL | PostgreSQL: Documentation: 18: 9... |
| **EXACT_02** | `Exact_Terminology` | Docker | Docker multi-stage builds Dockerfile CO... | 63.16 | **36.93** | Docker | Multi-stage builds | Docker Docs |
| **EXACT_03** | `Exact_Terminology` | PyTorch | PyTorch torch.distributed.fsdp.fully_sh... | 84.45 | **38.00** | PyTorch | torch.distributed.fsdp.fully_sha... |
| **EXACT_04** | `Exact_Terminology` | Linux Kernel | Linux Kernel /proc/sys/vm/drop_caches s... | 76.00 | **31.54** | Linux Kernel | Documentation for /proc/sys/vm/ ... |
| **EXACT_05** | `Exact_Terminology` | FastAPI | FastAPI dependency injection Depends() ... | 43.08 | **24.71** | FastAPI | Dependencies - FastAPI |
| **TYPO_01** | `Spelling_Error` | Kubernetes | kubernets taints and tolerations NoSche... | 60.67 | **34.33** | Kubernetes | Taints and Tolerations | Kubernetes |
| **TYPO_02** | `Spelling_Error` | PostgreSQL | postgresql jsonb contanment operater GI... | 35.94 | **23.89** | PostgreSQL | PostgreSQL: Documentation: 16: 8... |
| **TYPO_03** | `Spelling_Error` | PyTorch | pytorh torch.compile troubleshoting gra... | 40.82 | **22.72** | Hugging Face | torch.compile · Hugging Face |
| **TYPO_04** | `Spelling_Error` | Docker | dockr multi-stage bulds Dockerfile opti... | 38.74 | **29.06** | Docker | Multi-stage builds | Docker Docs |
| **TYPO_05** | `Spelling_Error` | RabbitMQ | rabitmq cluster formation and peer disc... | 20.10 | **22.47** | RabbitMQ | Cluster Formation and Peer Disco... |
| **SHORT_KW_01** | `Short_Keyword` | Docker | docker compose | 14.98 | **8.53** | Docker | Install the Docker Compose stand... |
| **SHORT_KW_02** | `Short_Keyword` | AWS | aws glue json | 21.30 | **12.27** | AWS | Using the JSON format in AWS Glu... |
| **SHORT_KW_03** | `Short_Keyword` | PyTorch | pytorch autograd function | 22.38 | **9.22** | PyTorch | Extending torch.func with autogr... |
| **SHORT_KW_09** | `Short_Keyword` | gRPC | grpc alts authentication | 23.60 | **13.51** | gRPC | ALTS authentication | Java | gRPC |
| **SHORT_KW_10** | `Short_Keyword` | React | react useEffect | 28.86 | **11.74** | OpenTelemetry | React Native App | OpenTelemetry |
| **SHORT_TYPO_01** | `Short_Spelling_Error` | Docker | dockr compse | 15.06 | **8.53** | Docker | Install the Docker Compose stand... |
| **SHORT_TYPO_02** | `Short_Spelling_Error` | AWS | aws opensearh | 5.88 | **3.33** | AWS | Using the AWS Marketplace API to... |
| **SHORT_TYPO_04** | `Short_Spelling_Error` | PyTorch | pytorh cuda memory | 13.13 | **9.74** | Hugging Face | Trainer · Hugging Face |
| **SHORT_TYPO_05** | `Short_Spelling_Error` | PostgreSQL | postgresql reindx | 6.61 | **10.09** | PostgreSQL | PostgreSQL: Documentation: 8.2: ... |
