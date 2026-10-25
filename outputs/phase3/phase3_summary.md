# Phase 3 — Classic IR Engine: BM25 vs. Hierarchical BM25F Results

**Engine Configuration:** Scorer Mode: `both`
**BM25 Parameters:** k1=1.2, b=0.75, top_k=20
**BM25F Zone Weights:** {'title': 4.0, 'heading': 2.5, 'code': 2.0, 'body': 1.0}
**Hierarchical Taxonomy Boost:** 25%
**Index Size:** 2078.0 MB
**Elapsed Time:** 4854.59s

## 1. Index Statistics & Hierarchical Postings

| Metric | Value |
| :--- | :--- |
| Total Documents | 77,565 |
| Vocabulary Size | 1,824,935 |
| Total Postings Entries | 18,821,101 |
| Total Position Entries | 95,481,935 |
| avg_postings_per_term | 10.31 |
| Hierarchical Categories | 13 |
| Hierarchical Technologies | 76 |
| Hierarchical Domains | 5 |
| Total Sections Indexed | 851,580 |
| avg_section_length | 79.6 |
| Champion Lists Count | 1,824,935 |

## 2. Average Zone Lengths across Corpus

| Zone | Average Tokens |
| :--- | :--- |
| **Title** | 7.6 tokens |
| **Heading** | 38.4 tokens |
| **Body** | 896.6 tokens |
| **Code** | 289.0 tokens |

## 3. Query Suite Comparative Results (BM25 vs. Hierarchical BM25F)

| Query ID | Category | Query Text | BM25 Top Score | BM25F Top Score | BM25F Top Tech / Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | Negation | Asynchronous Python web frameworks that do NO... | `44.4596` | **`25.2862`** | Python (Prog Lang) |
| **Q02** | Negation | Kubernetes ingress controllers that support S... | `46.7089` | **`21.7763`** | Istio (Containers & Orchestration) |
| **Q03** | Negation | Java database connection pooling libraries th... | `46.5317` | **`21.1383`** | Spring Boot (Web & Backend Frameworks) |
| **Q04** | Homonyms | Python memory management reference counting v... | `31.1473` | **`14.7898`** | Python (Prog Lang) |
| **Q05** | Homonyms | Java thread synchronization primitives and mo... | `21.8364` | **`12.3864`** | Java (Prog Lang) |
| **Q06** | Homonyms | Docker container networking bridge mode confi... | `28.1469` | **`16.0840`** | Docker (Containers & Orchestration) |
| **Q07** | Homonyms | PostgreSQL connection pooling versus session ... | `40.7232` | **`19.6292`** | PostgreSQL (DB) |
| **Q08** | Temporal | React state management patterns introduced st... | `35.4599` | **`16.3479`** | React (Web & Backend Frameworks) |
| **Q09** | Temporal | Kubernetes security feature deprecations betw... | `43.5779` | **`25.1268`** | Kubernetes (Containers & Orchestration) |
| **Q10** | Temporal | Java module system additions and breaking cha... | `25.3554` | **`13.7417`** | Java (Prog Lang) |
| **Q11** | Temporal | FastAPI dependency injection updates in versi... | `31.7610` | **`18.2296`** | FastAPI (Web & Backend Frameworks) |
| **Q12** | Exact_Terminology | Handling ERR_CONNECTION_RESET exceptions in N... | `48.1977` | **`28.9008`** | Node.js (Web & Backend Frameworks) |
| **Q13** | Exact_Terminology | PostgreSQL JSONB containment operator @> perf... | `50.3380` | **`29.3950`** | PostgreSQL (DB) |
| **Q14** | Exact_Terminology | Docker multi-stage build caching syntax using... | `62.5152` | **`37.8129`** | Docker (Containers & Orchestration) |
| **Q15** | Exact_Terminology | Configuring PyTorch distributed data parallel... | `44.2117` | **`24.7002`** | PyTorch (AI/ML) |
| **Q16** | Multi_Constraint | How to configure zero-downtime rolling update... | `48.5785` | **`22.4263`** | AWS (Cloud) |
| **Q17** | Multi_Constraint | FastAPI background tasks integration with Red... | `37.3876` | **`21.5393`** | FastAPI (Web & Backend Frameworks) |
| **Q18** | Multi_Constraint | Spring Boot actuator endpoint security config... | `55.1923` | **`31.6138`** | Spring Boot (Web & Backend Frameworks) |
| **Q19** | Multi_Constraint | Terraform AWS provider state locking configur... | `69.5326` | **`40.5711`** | Terraform (DevOps & CI/CD) |
| **Q20** | Multi_Constraint | Prometheus scraping metrics from secure gRPC ... | `57.9923` | **`31.7428`** | Grafana (Observability & Monitoring) |
| **Q21** | Error_Troubleshooting | Resolving org.apache.kafka.common.errors.Retr... | `38.5116` | **`17.5312`** | Elasticsearch (DB) |
| **Q22** | Error_Troubleshooting | Fixing Docker buildkit out of memory error du... | `41.3281` | **`23.6447`** | Docker (Containers & Orchestration) |
| **Q23** | Configuration_Schema | Kubernetes deployment resource limits CPU and... | `30.7697` | **`18.1194`** | Kubernetes (Containers & Orchestration) |
| **Q24** | Configuration_Schema | Terraform AWS security group ingress block CI... | `48.6729` | **`28.9308`** | Terraform (DevOps & CI/CD) |
| **Q25** | Conceptual_Procedural | Architecture difference between Apache Kafka ... | `70.3423` | **`40.1382`** | Pulsar (Communication/msg passing) |
