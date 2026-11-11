# Phase 3 — Classic IR Engine: BM25 vs. Hierarchical BM25F Results

**Engine Configuration:** Scorer Mode: `both`
**BM25 Parameters:** k1=1.2, b=0.75, top_k=20
**BM25F Zone Weights:** {'title': 4.0, 'heading': 2.5, 'code': 2.0, 'body': 1.0}
**Hierarchical Taxonomy Boost:** 25%
**Index Size:** 1606.37 MB
**Elapsed Time:** 60.17s

## 1. Index Statistics & Hierarchical Postings

| Metric | Value |
| :--- | :--- |
| Total Documents | 61,966 |
| Vocabulary Size | 1,799,233 |
| Total Postings Entries | 14,338,715 |
| Total Position Entries | 72,795,150 |
| avg_postings_per_term | 7.97 |
| Hierarchical Categories | 13 |
| Hierarchical Technologies | 76 |
| Hierarchical Domains | 5 |
| Total Sections Indexed | 652,089 |
| avg_section_length | 79.6 |
| Champion Lists Count | 1,799,233 |

## 2. Average Zone Lengths across Corpus

| Zone | Average Tokens |
| :--- | :--- |
| **Title** | 7.7 tokens |
| **Heading** | 36.9 tokens |
| **Body** | 840.5 tokens |
| **Code** | 290.4 tokens |

## 3. Query Suite Comparative Results (BM25 vs. Hierarchical BM25F)

| Query ID | Category | Query Text | BM25 Top Score | BM25F Top Score | BM25F Top Tech / Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | Negation | Manage TLS certificates in MongoDB Kubernetes... | `72.3692` | **`43.6837`** | MongoDB (DB) |
| **Q02** | Negation | Configure Istio Ingress Gateway for HTTPS tra... | `59.6681` | **`35.0138`** | Istio (Containers & Orchestration) |
| **Q03** | Negation | Execute SQL statements in Go database/sql tha... | `57.4313` | **`33.2099`** | Go (Prog Lang) |
| **Q04** | Homonyms | Go garbage collector tracing cycle and memory... | `53.7475` | **`31.7295`** | Go (Prog Lang) |
| **Q05** | Homonyms | Rust thread-safe reference-counting pointer A... | `83.6764` | **`49.1699`** | Rust (Prog Lang) |
| **Q06** | Homonyms | Docker container host network driver mode and... | `44.2045` | **`25.2164`** | Docker (Containers & Orchestration) |
| **Q07** | Homonyms | Redis Hash vs JSON storage data structures an... | `37.9224` | **`21.7789`** | Redis (DB) |
| **Q08** | Temporal | React 19 useEffectEvent hook for extracting n... | `77.3637` | **`33.1674`** | React (Web & Backend Frameworks) |
| **Q09** | Temporal | Kubernetes removed feature gates and deprecat... | `35.9118` | **`20.2559`** | Kubernetes (Containers & Orchestration) |
| **Q10** | Temporal | Prometheus 3.0 migration guide metric_name_va... | `57.3157` | **`33.9521`** | Prometheus (Observability & Monitoring) |
| **Q11** | Temporal | Go encoding/json/v2 migration guide API and b... | `38.4945` | **`22.4916`** | Go (Prog Lang) |
| **Q12** | Exact_Terminology | PostgreSQL pg_mcv_list_items statistics infor... | `110.3315` | **`63.2147`** | PostgreSQL (DB) |
| **Q13** | Exact_Terminology | PostgreSQL JSONB containment operator @> and ... | `81.4851` | **`47.4540`** | PostgreSQL (DB) |
| **Q14** | Exact_Terminology | Linux kernel /proc/sys/vm/drop_caches sysctl ... | `86.3803` | **`34.9028`** | Linux Kernel (OS, Networking & Web Servers) |
| **Q15** | Exact_Terminology | PyTorch torch.distributed.fsdp.fully_shard FS... | `122.3317` | **`52.8822`** | PyTorch (AI/ML) |
| **Q16** | Multi_Constraint | Perform a rolling update on a Kubernetes Daem... | `92.6438` | **`53.4920`** | Kubernetes (Containers & Orchestration) |
| **Q17** | Multi_Constraint | FastAPI BackgroundTasks asynchronous email no... | `72.3936` | **`41.4287`** | FastAPI (Web & Backend Frameworks) |
| **Q18** | Multi_Constraint | HashiCorp Vault cluster restore from Raft sna... | `93.9770` | **`56.2869`** | Vault (Security & Identity) |
| **Q19** | Multi_Constraint | Terraform Amazon S3 backend state storage con... | `75.6041` | **`43.1171`** | Terraform (DevOps & CI/CD) |
| **Q20** | Multi_Constraint | Monitoring Docker container metrics using cAd... | `94.8310` | **`54.7109`** | Prometheus (Observability & Monitoring) |
| **Q21** | Error_Troubleshooting | PyTorch torch.compile troubleshooting graph b... | `84.0954` | **`36.9048`** | Hugging Face (AI/ML) |
| **Q22** | Error_Troubleshooting | Istio sidecar injection problems troubleshoot... | `58.8543` | **`34.7181`** | Istio (Containers & Orchestration) |
| **Q23** | Configuration_Schema | Kubernetes Taints and Tolerations kubectl tai... | `85.0276` | **`48.3410`** | Kubernetes (Containers & Orchestration) |
| **Q24** | Configuration_Schema | ClickHouse ALTER TABLE DETACH PARTITION distr... | `51.6329` | **`30.2405`** | ClickHouse (DB) |
| **Q25** | Conceptual_Procedural | RabbitMQ consumer delivery acknowledgements a... | `129.6514` | **`74.9592`** | RabbitMQ (Communication/msg passing) |
