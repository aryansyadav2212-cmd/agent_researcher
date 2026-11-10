from fastapi import FastAPI
from .client import router

app = FastAPI()

app.include_router(router)