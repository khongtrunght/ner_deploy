import uvicorn
from fastapi import FastAPI
from logic import get_annotation, get_annotation_lightning

app = FastAPI()

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
print("Ready to run ", get_annotation_lightning("Bác sĩ đi xe ô tô làm việc tại Bênh viện Bạch Mai"))

@app.post("/ner")
def do_ner(text: str):
    return {"text": get_annotation(text)}


@app.post("/ner_lightning")
def do_ner_lightning(text: str):
    return {"text": get_annotation_lightning(text)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
