# Test Queries Benchmark — Classical IR Retrieval Results

- **Queries Evaluated**: 34 across 5 Query Types (Natural Language, Short Keyword, Boolean, Exact Terminology, Spelling Errors)
- **Index Documents**: 77,565
- **Index Vocabulary**: 1,824,935 unique stemmed terms
- **BM25 Execution Time**: 2.72s (80.1ms/q)
- **BM25F Execution Time**: 2.89s (85.0ms/q)
- **Multi-Level Execution Time**: 3.92s (115.2ms/q)

## Performance By Query Type & Category

| Query ID | Type | Target Tech | Query Text | BM25 Top Score | BM25F Top Score | Top Retrieved Tech (BM25F) | Top Match Zone/Title |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NL_01** | `Natural_Language` | PyTorch | How to handle out of memory errors when... | 50.26 | **25.32** | Ray | Ray Examples — Ray 2.58.0 |
| **NL_02** | `Natural_Language` | Kubernetes | Configuring zero downtime rolling updat... | 37.53 | **21.49** | Kubernetes | Horizontal Pod Autoscaling | Kub... |
| **NL_03** | `Natural_Language` | PostgreSQL | Best practices for indexing JSONB docum... | 39.04 | **22.75** | PostgreSQL | PostgreSQL: Documentation: 18: 8... |
| **NL_04** | `Natural_Language` | Python | How does reference counting interact wi... | 48.03 | **26.16** | Python | Glossary — Python 3.14.7 documen... |
| **NL_05** | `Natural_Language` | FastAPI | Setting up asynchronous background task... | 41.91 | **22.33** | FastAPI | Background Tasks - FastAPI |
| **KW_01** | `Keyword` | Docker | docker compose healthcheck interval tim... | 33.53 | **18.93** | Docker | Control startup and shutdown ord... |
| **KW_02** | `Keyword` | Kafka | kafka consumer group rebalance listener | 31.17 | **15.83** | Pulsar | Pulsar adaptor for Apache Kafka ... |
| **KW_03** | `Keyword` | Redis | redis pub sub message loss pattern | 29.59 | **17.01** | Redis | Redis pub/sub with redis-rb | Docs |
| **KW_04** | `Keyword` | Prometheus | prometheus scrape configs tls cert secret | 37.30 | **20.31** | Prometheus | Configuration | Prometheus |
| **KW_05** | `Keyword` | Rust | rust borrow checker mutable reference l... | 42.28 | **24.87** | Rust | Validating References with Lifet... |
| **BOOL_01** | `Boolean` | Python | python web framework AND asynchronous N... | 34.60 | **19.75** | Python | asyncio — Asynchronous I/O — Pyt... |
| **BOOL_02** | `Boolean` | Kubernetes | kubernetes ingress controller AND SSL p... | 43.51 | **20.14** | Istio | Istio / Ingress Gateway without ... |
| **BOOL_03** | `Boolean` | Java | database connection pooling AND java NO... | 28.51 | **13.06** | Spring Boot | 32. Working with SQL Databases |
| **BOOL_04** | `Boolean` | Terraform | terraform state locking AND (dynamodb O... | 39.15 | **23.25** | Terraform | Python Reference for Structs | T... |
| **BOOL_05** | `Boolean` | Docker | docker container networking AND (bridge... | 33.81 | **18.98** | Docker | Networking in Compose | Docker Docs |
| **EXACT_01** | `Exact_Terminology` | PostgreSQL | PostgreSQL JSONB containment operator @> | 48.19 | **28.19** | PostgreSQL | PostgreSQL: Documentation: 18: 8... |
| **EXACT_02** | `Exact_Terminology` | Docker | Docker multi-stage build caching syntax... | 61.82 | **37.39** | Docker | Building best practices | Docker... |
| **EXACT_03** | `Exact_Terminology` | PyTorch | PyTorch find_unused_parameters=True Dis... | 49.58 | **29.49** | PyTorch | Distributed Optimizers — PyTorch... |
| **EXACT_04** | `Exact_Terminology` | Kafka | Resolving org.apache.kafka.common.error... | 34.93 | **15.62** | Elasticsearch | Kafka output plugin v10.6.0 | Lo... |
| **EXACT_05** | `Exact_Terminology` | FastAPI | FastAPI dependency injection updates in... | 31.76 | **18.23** | FastAPI | Dependencies - FastAPI |
| **TYPO_01** | `Spelling_Error` | Kubernetes | kubernets deploymnt rediness probe conf... | 20.57 | **9.27** | Redis | Redis Enterprise for Kubernetes ... |
| **TYPO_02** | `Spelling_Error` | PostgreSQL | postgre sql jsonb contanment operater p... | 23.50 | **13.66** | PostgreSQL | PostgreSQL: Documentation: 18: 8... |
| **TYPO_03** | `Spelling_Error` | PyTorch | pytorh cuda out of memry eror trainig | 14.80 | **8.03** | React | renderToReadableStream – React |
| **TYPO_04** | `Spelling_Error` | Docker | dockr contaner netwrking brigde mode | 21.00 | **10.25** | RFCs | www.rfc-editor.org_rfc_rfc559.html |
| **TYPO_05** | `Spelling_Error` | Kafka | apche kafak consmer rebalence retriable... | 15.11 | **6.83** | Azure | Service discovery resiliency (pr... |
| **SHORT_KW_01** | `Short_Keyword` | Docker | docker compose | 14.82 | **8.43** | Docker | Install the Docker Compose stand... |
| **SHORT_KW_02** | `Short_Keyword` | AWS | aws kafka | 15.80 | **9.03** | AWS | 了解服務執行角色 - Amazon Managed Stream... |
| **SHORT_KW_03** | `Short_Keyword` | PyTorch | pytorch ddp | 20.30 | **11.47** | PyTorch | DDP Communication Hooks — PyTorc... |
| **SHORT_KW_09** | `Short_Keyword` | gRPC | grpc tls | 14.64 | **7.41** | Envoy | Secret discovery service (SDS) —... |
| **SHORT_KW_10** | `Short_Keyword` | React | react hooks | 16.92 | **9.62** | React | React 调用组件和 Hook – React 中文文档 |
| **SHORT_TYPO_01** | `Short_Spelling_Error` | Docker | dockr compse | 15.60 | **8.84** | Linux Kernel | Net DIM - Generic Network Dynami... |
| **SHORT_TYPO_02** | `Short_Spelling_Error` | AWS | aws kafak | 6.14 | **3.49** | AWS | AWS 리전에서 AWS STS 관리 - AWS Identi... |
| **SHORT_TYPO_04** | `Short_Spelling_Error` | PyTorch | pytorh cuda | 8.12 | **4.62** | CUDA | cuTENSOR Functions — cuTENSOR 1.... |
| **SHORT_TYPO_05** | `Short_Spelling_Error` | PostgreSQL | postgre jsnb | 6.43 | **2.93** | Docker | PostgreSQL specific guide | Dock... |
