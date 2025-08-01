from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI(title="HackRx 6.0 API")

@app.get("/")
def read_root():
    return {"message": "Welcome to HackRx 6.0 Intelligent Query-Retrieval System"}

@app.get("/health")
def health_check():
    return {"status": "ok"}