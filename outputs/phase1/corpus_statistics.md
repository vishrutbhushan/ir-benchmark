# Phase 1 — Structured Corpus Statistics Report

**Total Structured Documents Processed:** 77,565
**Average Document Length:** 1,058.2 words (7,851.6 characters)
**Total Vocabulary / Body Words:** 82,082,583

## Visual Charts & Graphs

![Phase 1 Category Distribution](charts/phase1_category_distribution.png)

![Phase 1 Code Prevalence](charts/phase1_code_prevalence.png)

![Phase 1 Top Technologies](charts/phase1_top_technologies.png)

## 1. Code Block & Technical Token Extraction

- **Documents with Code Blocks:** 49,857 (64.28%)
- **Total Isolated Code Blocks:** 480,280
- **Average Code Blocks per Document:** 6.19
- **Total Code Snippet Characters:** 143,622,372

## 2. Date Detection & Normalization

- **Documents with Dates Detected:** 10,407 (13.42%)
- **Total Normalized Dates Extracted:** 23,496

### Top Normalized Dates Detected

| Date (ISO) | Document Frequency |
| :--- | :--- |
| `2026-08-13` | 2,345 |
| `2025-11-17` | 736 |
| `2026-04-20` | 608 |
| `2023-11-20` | 580 |
| `2026-06-29` | 562 |
| `2025-08-18` | 508 |
| `2026-02-02` | 455 |
| `2025-06-16` | 430 |

## 3. Category Breakdown

| Category | Documents | Share |
| :--- | :--- | :--- |
| **programming-language** | 3,785 | 4.9% |
| **framework** | 6,039 | 7.8% |
| **containers-orchestration** | 2,836 | 3.7% |
| **distributed-data** | 6,303 | 8.1% |
| **database-search** | 17,451 | 22.5% |
| **cloud** | 12,217 | 15.8% |
| **devops-infrastructure** | 10,621 | 13.7% |
| **os-networking-web** | 7,608 | 9.8% |
| **observability** | 3,236 | 4.2% |
| **security** | 62 | 0.1% |
| **ml-infrastructure** | 7,407 | 9.5% |

## 4. Technology Breakdown (Top 20 by Volume)

| Technology | Docs | Avg Words | Avg Chars | Code Blocks | Dates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Django | 2,440 | 1311.9 | 9690.4 | 17,982 | 883 |
| PostgreSQL | 2,440 | 918.4 | 5844.5 | 7,286 | 2736 |
| GitHub | 2,434 | 992.2 | 7343.4 | 11,330 | 522 |
| Java | 2,431 | 628.7 | 6348.3 | 17,899 | 13 |
| Azure | 2,431 | 2045.5 | 14216.7 | 15,950 | 869 |
| Elasticsearch | 2,427 | 844.2 | 6297.3 | 9,624 | 242 |
| Grafana | 2,422 | 738.2 | 5323.4 | 8,963 | 243 |
| Redis | 2,418 | 1048.5 | 7665.4 | 20,671 | 47 |
| ClickHouse | 2,417 | 572.9 | 4473.3 | 25,759 | 461 |
| Envoy | 2,414 | 467.3 | 3552.0 | 5,861 | 805 |
| MySQL | 2,408 | 719.4 | 5613.2 | 6,320 | 268 |
| Cloudflare | 2,400 | 685.7 | 4967.6 | 7,285 | 112 |
| Spark | 2,393 | 664.1 | 6095.4 | 14,181 | 198 |
| Oracle Cloud | 2,391 | 485.7 | 4797.7 | 21,603 | 98 |
| Hadoop | 2,388 | 529.9 | 4683.6 | 16,931 | 105 |
| Linux Kernel | 2,387 | 1362.9 | 9379.8 | 9,717 | 99 |
| AWS | 2,373 | 653.4 | 5376.0 | 10,967 | 230 |
| OpenSearch | 2,309 | 831.0 | 6252.2 | 16,999 | 117 |
| CUDA | 2,295 | 1650.5 | 12026.8 | 10,391 | 177 |
| MongoDB | 2,231 | 705.9 | 5059.0 | 11,146 | 639 |