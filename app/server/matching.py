#%%
print()
#%%
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from inference.linker import EntityLinker

access_token = 'hf_......'

app = Flask(__name__)
# Cross-origin requests allowed
cors = CORS(app, resources={r"/*": {"origins": "*"}})

dict_occupations = pd.read_csv("occupations_en.csv", sep=",", header=0)
#preferredLabel
#code

#%%
@app.route("/match", methods=["POST"])
def match():
    job_descr = request.form["job_descr"]
    custom_pipeline = EntityLinker(entity_model = 'tabiya/bert-base-job-extract', similarity_model = 'all-MiniLM-L6-v2', hf_token = access_token)
    extracted = custom_pipeline(job_descr)
    print(extracted)
    for elem in extracted:
        if elem['type'] == "Occupation":
            new_list = []
            for esco_code in elem['retrieved']:
                row = dict_occupations[dict_occupations['code']==esco_code]
                preferredLabel = row['preferredLabel'].iloc[0]
                conceptUri = row['conceptUri'].iloc[0]
                new_list = new_list + [{'code':esco_code, 'uri': conceptUri, 'label':preferredLabel}]
            elem['retrieved'] = new_list
    return extracted

if __name__ == "__main__":
    app.run(host="", port=5000, debug=True)
