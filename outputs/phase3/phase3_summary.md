# Phase 3 — Classic IR Engine: BM25 vs. Hierarchical BM25F Results

**Engine Configuration:** Scorer Mode: `both`
**BM25 Parameters:** k1=1.2, b=0.75, top_k=20
**BM25F Zone Weights:** {'title': 4.0, 'heading': 2.5, 'code': 2.0, 'body': 1.0}
**Hierarchical Taxonomy Boost:** 25%
**Index Size:** 0.62 MB
**Elapsed Time:** 0.12s

## 1. Index Statistics & Hierarchical Postings

| Metric | Value |
| :--- | :--- |
| Total Documents | 10 |
| Vocabulary Size | 3,374 |
| Total Postings Entries | 4,096 |
| Total Position Entries | 23,745 |
| avg_postings_per_term | 1.21 |
| Hierarchical Categories | 6 |
| Hierarchical Technologies | 9 |
| Hierarchical Domains | 5 |

## 2. Average Zone Lengths across Corpus

| Zone | Average Tokens |
| :--- | :--- |
| **Title** | 5.5 tokens |
| **Heading** | 145.1 tokens |
| **Body** | 2052.6 tokens |
| **Code** | 171.6 tokens |

## 3. Query Suite Comparative Results (BM25 vs. Hierarchical BM25F)

| Query ID | Category | Query Text | BM25 Top Score | BM25F Top Score | BM25F Top Tech / Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | Negation | Asynchronous Python web frameworks that do NO... | `5.0462` | **`2.3676`** | RFCs (OS, Networking & Web Servers) |
| **Q02** | Negation | Kubernetes ingress controllers that support S... | `8.7057` | **`4.0169`** | Redis (DB) |
| **Q03** | Negation | Java database connection pooling libraries th... | `3.8564` | **`2.2198`** | Java (Prog Lang) |
| **Q04** | Homonyms | Python memory management reference counting v... | `8.0489` | **`3.5989`** | RFCs (OS, Networking & Web Servers) |
| **Q05** | Homonyms | Java thread synchronization primitives and mo... | `3.8564` | **`2.2198`** | Java (Prog Lang) |
| **Q06** | Homonyms | Docker container networking bridge mode confi... | `7.0532` | **`3.3265`** | Kubeflow (ML Ops) |
| **Q07** | Homonyms | PostgreSQL connection pooling versus session ... | `2.2720` | **`1.0466`** | React (Web & Backend Frameworks) |
| **Q08** | Temporal | React state management patterns introduced st... | `8.4598` | **`4.8486`** | React (Web & Backend Frameworks) |
| **Q09** | Temporal | Kubernetes security feature deprecations betw... | `10.5182` | **`4.8816`** | Redis (DB) |
| **Q10** | Temporal | Java module system additions and breaking cha... | `8.3731` | **`3.8778`** | RFCs (OS, Networking & Web Servers) |
| **Q11** | Temporal | FastAPI dependency injection updates in versi... | `5.3543` | **`2.4406`** | Redis (DB) |
| **Q12** | Exact_Terminology | Handling ERR_CONNECTION_RESET exceptions in N... | `12.6716` | **`7.1436`** | RFCs (OS, Networking & Web Servers) |
| **Q13** | Exact_Terminology | PostgreSQL JSONB containment operator @> perf... | `4.6898` | **`2.1062`** | Kubeflow (ML Ops) |
| **Q14** | Exact_Terminology | Docker multi-stage build caching syntax using... | `6.1652` | **`3.2726`** | RFCs (OS, Networking & Web Servers) |
| **Q15** | Exact_Terminology | Configuring PyTorch distributed data parallel... | `26.8072` | **`12.3402`** | Kubeflow (ML Ops) |
| **Q16** | Multi_Constraint | How to configure zero-downtime rolling update... | `14.2270` | **`6.5085`** | Redis (DB) |
| **Q17** | Multi_Constraint | FastAPI background tasks integration with Red... | `15.9153` | **`9.1887`** | Redis (DB) |
| **Q18** | Multi_Constraint | Spring Boot actuator endpoint security config... | `13.7908` | **`6.4545`** | Redis (DB) |
| **Q19** | Multi_Constraint | Terraform AWS provider state locking configur... | `6.1113` | **`2.8068`** | Rust (Prog Lang) |
| **Q20** | Multi_Constraint | Prometheus scraping metrics from secure gRPC ... | `8.5990` | **`4.0237`** | Redis (DB) |
| **Q21** | Error_Troubleshooting | Resolving org.apache.kafka.common.errors.Retr... | `5.6850` | **`2.8330`** | RFCs (OS, Networking & Web Servers) |
| **Q22** | Error_Troubleshooting | Fixing Docker buildkit out of memory error du... | `6.4896` | **`3.0060`** | Kubeflow (ML Ops) |
| **Q23** | Configuration_Schema | Kubernetes deployment resource limits CPU and... | `11.0415` | **`5.1171`** | Redis (DB) |
| **Q24** | Configuration_Schema | Terraform AWS security group ingress block CI... | `12.0185` | **`5.6409`** | RFCs (OS, Networking & Web Servers) |
| **Q25** | Conceptual_Procedural | Architecture difference between Apache Kafka ... | `10.8900` | **`5.4475`** | RFCs (OS, Networking & Web Servers) |
