from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.pipeline import run_pipeline
from app.schemas import CodeRequest


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run")
def run(request: CodeRequest):
    return run_pipeline(
        code=request.code,
        language=request.language,
        mode=request.mode,
        user_query=request.user_query,
        user_id=request.user_id,
        intent=request.intent,
    )
