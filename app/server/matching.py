# %%
print()
# %%
import sys
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference.linker import EntityLinker

# Add the parent and the current directory to the path
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load occupations data
try:
    dict_occupations = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "occupations_en.csv")), sep=",", header=0)
except FileNotFoundError:
    print("Error: occupations_en.csv file not found.")
    sys.exit(1)

custom_pipeline = EntityLinker(entity_model='tabiya/bert-base-job-extract', similarity_model='all-MiniLM-L6-v2', output_format='all', k=10)
# preferredLabel
# code

# %%

@app.route('/', methods=['GET'])
def index():
    return render_template('client.html')

@app.route("/match", methods=["POST"])
def match():
    job_descr = request.form.get("job_descr")
    if not job_descr:
        return jsonify({"error": "job_descr is required"}), 400

    extracted = custom_pipeline(job_descr)
    if not extracted:
        return jsonify({"error": "No entities extracted"}), 400

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
            # get the list of skills from the list of retrieved objects
            new_list = [retrieved.skills for retrieved in elem['retrieved']]
            elem['retrieved'] = new_list
        elif elem['type'] == "Qualification":
            # get the list of qualifications from the list of retrieved objects
            new_list = [f"{retrieved.qualification}: EQF level {int(retrieved.eqf_level)}" for retrieved in elem['retrieved']]
            # remove duplicates
            new_list = list(dict.fromkeys(new_list))
            elem['retrieved'] = new_list
        else:
            elem['retrieved'] = ["Type not recognized"]
    
    return jsonify(extracted)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)