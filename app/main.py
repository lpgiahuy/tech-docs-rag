from fastapi import FastAPI

app = FastAPI(title="tech-docs-rag")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
