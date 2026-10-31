"""Deterministic URL-pattern quality filters for the shared corpus."""

import re
from urllib.parse import urlsplit

BLOCKED_DOMAINS = frozenset(
    {
        "mongodbcom-cdn.staging.corp.mongodb.com",
        "origin.kernel.org",
        "eui.elastic.co",
        "dell.huggingface.co",
        "blog.rabbitmq.com",
        "blog.gradle.org",
        "slack.linkerd.io",
        "www.rfc-editor.org",
        "csrc.nist.gov",
    }
)
BLOCKED_TECHNOLOGIES = frozenset({"RFCs", "NIST"})
_OPENSSL_MAN3 = re.compile(r"^docs\.openssl\.org$")
_MAN3_PATH = re.compile(r"/man3/")
_JAVADOC_STUB_FILENAME = re.compile(
    r"(?:package-(?:summary|tree|use|frame|list|detail)"
    r"|overview-(?:summary|tree)|allclasses(?:-frame|-noframe)?"
    r"|allpackages|allmodules|module-summary|serialized-form"
    r"|deprecated-list|constant-values|help-doc)\.html$",
    re.IGNORECASE,
)
_JAVADOC_API_PATH = re.compile(
    r"/api/(?:org|com|java|javax|jdk|sun|io)/[a-z]", re.IGNORECASE
)
_CLASS_USE = re.compile(r"/class-use/", re.IGNORECASE)
_EXTERNAL_REDIRECT = re.compile(r"[?&]is-external=true")
_GRADLE_KOTLIN_DSL = re.compile(r"/kotlin-dsl/")
_CF_REST_API = re.compile(r"/api/(?:typescript|resources)/")
_ELASTIC_REST_API = re.compile(r"/api/doc/(?:kibana|elasticsearch|serverless)/")
_CUDA_ARCHIVE = re.compile(r"/cuda/archive/")
_DJANGO_RELEASE = re.compile(r"/releases/\d+\.\d+")
_SPRING_JAVADOC = re.compile(
    r"/spring-boot/docs/\d+\.\d+.*?/api/(?:org|com)/", re.IGNORECASE
)
_SEARCH_PAGINATION = re.compile(r"[?&]p(?:age)?=\d+")


def quality_filter_reason(record: dict) -> str | None:
    """Return a rejection reason, or ``None`` when the record passes."""
    parsed = urlsplit(record.get("url", ""))
    netloc = parsed.netloc
    path = parsed.path
    filename = path.rstrip("/").rsplit("/", 1)[-1].lower()
    technology = record.get("technology", "")

    if netloc in BLOCKED_DOMAINS:
        return f"blocked_domain:{netloc}"
    if technology in BLOCKED_TECHNOLOGIES:
        return f"blocked_technology:{technology}"
    if _OPENSSL_MAN3.match(netloc) and _MAN3_PATH.search(path):
        return "openssl_man3_api"
    if _JAVADOC_STUB_FILENAME.search(filename):
        return "javadoc_stub_filename"
    if _JAVADOC_API_PATH.search(path):
        return "javadoc_api_path"
    if _CLASS_USE.search(path):
        return "javadoc_class_use"
    if _EXTERNAL_REDIRECT.search(parsed.query):
        return "javadoc_external_redirect"
    if technology == "Gradle" and _GRADLE_KOTLIN_DSL.search(path):
        return "gradle_kotlin_dsl_ref"
    if technology == "Cloudflare" and _CF_REST_API.search(path):
        return "cloudflare_rest_api_endpoint"
    if technology in ("Elasticsearch", "Kibana") and _ELASTIC_REST_API.search(path):
        return "elasticsearch_rest_api_endpoint"
    if technology == "CUDA" and _CUDA_ARCHIVE.search(path):
        return "cuda_archive_doc"
    if technology == "Django" and _DJANGO_RELEASE.search(path):
        return "django_per_version_release_notes"
    if technology == "Spring Boot" and _SPRING_JAVADOC.search(path):
        return "spring_boot_versioned_javadoc"
    if _SEARCH_PAGINATION.search(parsed.query):
        return "search_pagination_page"
    return None
