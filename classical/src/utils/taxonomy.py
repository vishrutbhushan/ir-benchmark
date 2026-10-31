"""
Taxonomy and Hierarchical Classification Mapping

Provides a 2-tier hierarchical taxonomy mapping all 76 technologies across the corpus
to specific Categories (matching Category_Labels.txt) and high-level Domains.
"""

from __future__ import annotations

import re
from typing import Any

# Technology to (Category, Domain, Aliases) mapping
TECHNOLOGY_TAXONOMY: dict[str, dict[str, Any]] = {
    # 1. Cloud
    "AWS": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["aws", "amazon web services", "ec2", "s3", "ecs", "lambda"]},
    "Azure": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["azure", "microsoft azure"]},
    "Google Cloud": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["gcp", "google cloud", "google cloud platform"]},
    "Cloudflare": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["cloudflare", "workers", "cdn"]},
    "DigitalOcean": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["digitalocean", "droplet"]},
    "Oracle Cloud": {"category": "Cloud", "domain": "Infrastructure", "aliases": ["oracle cloud", "oci"]},

    # 2. AI/ML
    "PyTorch": {"category": "AI/ML", "domain": "AI & Machine Learning", "aliases": ["pytorch", "torch", "nn", "autograd"]},
    "TensorFlow": {"category": "AI/ML", "domain": "AI & Machine Learning", "aliases": ["tensorflow", "tf", "keras"]},
    "Hugging Face": {"category": "AI/ML", "domain": "AI & Machine Learning", "aliases": ["hugging face", "huggingface", "transformers", "datasets"]},
    "CUDA": {"category": "AI/ML", "domain": "AI & Machine Learning", "aliases": ["cuda", "nvidia", "gpu"]},
    "Ray": {"category": "AI/ML", "domain": "AI & Machine Learning", "aliases": ["ray", "rllib", "ray tune"]},

    # 3. Prog Lang
    "Python": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["python", "cpython", "asyncio"]},
    "Java": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["java", "jvm", "jdk"]},
    "Go": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["go", "golang", "goroutine"]},
    "Rust": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["rust", "cargo", "borrow checker"]},
    "TypeScript": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["typescript", "ts"]},
    "JavaScript": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["javascript", "js", "ecmascript", "es6"]},
    "Kotlin": {"category": "Prog Lang", "domain": "Software Engineering", "aliases": ["kotlin", "coroutines"]},

    # 4. DB
    "PostgreSQL": {"category": "DB", "domain": "Data Systems", "aliases": ["postgresql", "postgres", "psql", "jsonb"]},
    "MySQL": {"category": "DB", "domain": "Data Systems", "aliases": ["mysql", "innodb"]},
    "MongoDB": {"category": "DB", "domain": "Data Systems", "aliases": ["mongodb", "mongo", "nosql"]},
    "Redis": {"category": "DB", "domain": "Data Systems", "aliases": ["redis", "in-memory", "cache"]},
    "Elasticsearch": {"category": "DB", "domain": "Data Systems", "aliases": ["elasticsearch", "elastic", "es"]},
    "OpenSearch": {"category": "DB", "domain": "Data Systems", "aliases": ["opensearch"]},
    "ClickHouse": {"category": "DB", "domain": "Data Systems", "aliases": ["clickhouse", "olap", "columnar"]},
    "Neo4j": {"category": "DB", "domain": "Data Systems", "aliases": ["neo4j", "cypher", "graph database"]},

    # 5. ML Ops
    "Kubeflow": {"category": "ML Ops", "domain": "AI & Machine Learning", "aliases": ["kubeflow", "kfp"]},
    "MLflow": {"category": "ML Ops", "domain": "AI & Machine Learning", "aliases": ["mlflow", "experiment tracking"]},

    # 6. Communication/msg passing
    "Kafka": {"category": "Communication/msg passing", "domain": "Data Systems", "aliases": ["kafka", "apache kafka", "consumer group", "broker"]},
    "Pulsar": {"category": "Communication/msg passing", "domain": "Data Systems", "aliases": ["pulsar", "apache pulsar", "bookkeeper"]},
    "RabbitMQ": {"category": "Communication/msg passing", "domain": "Data Systems", "aliases": ["rabbitmq", "amqp", "message broker"]},
    "ZooKeeper": {"category": "Communication/msg passing", "domain": "Data Systems", "aliases": ["zookeeper", "zk"]},
    "gRPC": {"category": "Communication/msg passing", "domain": "Software Engineering", "aliases": ["grpc", "protobuf", "protocol buffers"]},

    # 7. Containers & Orchestration
    "Kubernetes": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["kubernetes", "k8s", "kubectl", "pod", "ingress"]},
    "Docker": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["docker", "dockerfile", "container", "compose"]},
    "Helm": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["helm", "charts"]},
    "Istio": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["istio", "service mesh"]},
    "Linkerd": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["linkerd"]},
    "Podman": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["podman"]},
    "OpenShift": {"category": "Containers & Orchestration", "domain": "Infrastructure", "aliases": ["openshift", "okd"]},

    # 8. DevOps & CI/CD
    "Terraform": {"category": "DevOps & CI/CD", "domain": "Infrastructure", "aliases": ["terraform", "tf", "hcl", "infrastructure as code"]},
    "Git": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["git", "version control"]},
    "GitHub": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["github", "github actions"]},
    "GitLab": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["gitlab", "gitlab ci"]},
    "Jenkins": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["jenkins", "jenkinsfile"]},
    "Gradle": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["gradle", "build.gradle"]},
    "Maven": {"category": "DevOps & CI/CD", "domain": "Software Engineering", "aliases": ["maven", "pom.xml"]},

    # 9. Web & Backend Frameworks
    "FastAPI": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["fastapi", "pydantic", "starlette", "uvicorn"]},
    "React": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["react", "jsx", "reactjs", "react-router"]},
    "Angular": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["angular", "typescript framework"]},
    "Django": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["django", "django orm"]},
    "Node.js": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["node.js", "nodejs", "npm"]},
    "Spring": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["spring", "spring framework"]},
    "Spring Boot": {"category": "Web & Backend Frameworks", "domain": "Software Engineering", "aliases": ["spring boot", "actuator"]},

    # 10. Observability & Monitoring
    "Prometheus": {"category": "Observability & Monitoring", "domain": "Operations & Reliability", "aliases": ["prometheus", "promql", "metrics"]},
    "Grafana": {"category": "Observability & Monitoring", "domain": "Operations & Reliability", "aliases": ["grafana", "dashboards"]},
    "OpenTelemetry": {"category": "Observability & Monitoring", "domain": "Operations & Reliability", "aliases": ["opentelemetry", "otel", "traces", "spans"]},
    "Jaeger": {"category": "Observability & Monitoring", "domain": "Operations & Reliability", "aliases": ["jaeger", "distributed tracing"]},

    # 11. Distributed Big Data
    "Spark": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["spark", "apache spark", "pyspark", "rdd"]},
    "Hadoop": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["hadoop", "hdfs", "mapreduce"]},
    "Apache Beam": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["apache beam", "beam"]},
    "Apache Iceberg": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["iceberg", "apache iceberg", "table format"]},
    "HBase": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["hbase", "apache hbase"]},
    "Hive": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["hive", "apache hive"]},
    "Cassandra": {"category": "Distributed Big Data", "domain": "Data Systems", "aliases": ["cassandra", "apache cassandra", "cql"]},

    # 12. OS, Networking & Web Servers
    "Linux Kernel": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["linux", "kernel", "syscall", "epoll"]},
    "Nginx": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["nginx", "reverse proxy"]},
    "Apache HTTP Server": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["apache", "httpd"]},
    "Envoy": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["envoy", "envoy proxy"]},
    "HAProxy": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["haproxy", "load balancer"]},
    "RFCs": {"category": "OS, Networking & Web Servers", "domain": "Infrastructure", "aliases": ["rfc", "ietf", "http/2", "tcp", "udp"]},
    "OpenSSL": {"category": "OS, Networking & Web Servers", "domain": "Operations & Reliability", "aliases": ["openssl", "tls", "ssl", "certificates"]},

    # 13. Security & Identity
    "Vault": {"category": "Security & Identity", "domain": "Operations & Reliability", "aliases": ["vault", "hashicorp vault", "secrets"]},
    "OWASP": {"category": "Security & Identity", "domain": "Operations & Reliability", "aliases": ["owasp", "vulnerability", "xss", "csrf"]},
    "NIST": {"category": "Security & Identity", "domain": "Operations & Reliability", "aliases": ["nist", "cybersecurity"]},
    "OpenID Connect": {"category": "Security & Identity", "domain": "Operations & Reliability", "aliases": ["oidc", "openid", "oauth2", "jwt"]},
}


def get_taxonomy_for_tech(tech_name: str) -> dict[str, str]:
    """Resolves technology to its category and domain."""
    normalized = tech_name.strip()
    if normalized in TECHNOLOGY_TAXONOMY:
        entry = TECHNOLOGY_TAXONOMY[normalized]
        return {
            "technology": normalized,
            "category": entry["category"],
            "domain": entry["domain"],
        }
    return {
        "technology": normalized,
        "category": "Other",
        "domain": "General",
    }


def infer_query_taxonomy(query_text: str) -> dict[str, list[str]]:
    """
    Infers relevant technologies, categories, and domains from query text
    using word boundary matching over aliases.
    """
    q_lower = query_text.lower()
    matched_techs: list[str] = []
    matched_categories: set[str] = set()
    matched_domains: set[str] = set()

    for tech, info in TECHNOLOGY_TAXONOMY.items():
        for alias in info["aliases"]:
            # Check with word boundaries
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, q_lower):
                matched_techs.append(tech)
                matched_categories.add(info["category"])
                matched_domains.add(info["domain"])
                break

    return {
        "matched_technologies": matched_techs,
        "matched_categories": sorted(list(matched_categories)),
        "matched_domains": sorted(list(matched_domains)),
    }
