import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference.linker import EntityLinker
from app.server.common import add_common_middleware

app = FastAPI(title="Job Matching UI")

add_common_middleware(app)

app.mount("/static", StaticFiles(directory=os.path.join(PATH, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(PATH, "templates"))

try:
    dict_occupations = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "occupations_en.csv")), sep=",", header=0)
except FileNotFoundError:
    print("Error: occupations_en.csv file not found.")
    sys.exit(1)

custom_pipeline = EntityLinker(entity_model='tabiya/bert-base-job-extract', similarity_model='all-MiniLM-L6-v2', output_format='all', k=10)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("client.html", {"request": request})

@app.post("/match")
async def match(job_descr: str = Form(...)):
    if not job_descr:
        return JSONResponse(content={"error": "job_descr is required"}, status_code=400)

    extracted = custom_pipeline(job_descr)
    if not extracted:
        return JSONResponse(content={"error": "No entities extracted"}, status_code=400)

    for elem in extracted:
        if elem['type'] == "Occupation":
            new_list = []
            for occupation in elem['retrieved']:
                row = dict_occupations[dict_occupations['code'] == occupation.esco_code]
                if not row.empty:
                    preferredLabel = row['preferredLabel'].iloc[0]
                    conceptUri = row['conceptUri'].iloc[0]
                    new_list.append({'code': occupation.esco_code, 'uri': conceptUri, 'label': preferredLabel})
            elem['retrieved'] = new_list
        elif elem['type'] == "Skill":
            new_list = [retrieved.skills for retrieved in elem['retrieved']]
            elem['retrieved'] = new_list
        elif elem['type'] == "Qualification":
            new_list = [f"{retrieved.qualification}: EQF level {int(retrieved.eqf_level)}" for retrieved in elem['retrieved']]
            new_list = list(dict.fromkeys(new_list))
            elem['retrieved'] = new_list
        else:
            elem['retrieved'] = ["Type not recognized"]

    return extracted

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server.matching:app", host="0.0.0.0", port=5001, log_level="info")
