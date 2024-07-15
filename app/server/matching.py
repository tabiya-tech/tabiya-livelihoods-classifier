# %%
print()
# %%
import sys
import os
import pandas as pd
from flask import Flask, request, send_from_directory, jsonify, render_template
from flask_cors import CORS
from inference.linker import EntityLinker

# Add the parent and the current directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load occupations data
try:
    dict_occupations = pd.read_csv("occupations_en.csv", sep=",", header=0)
except FileNotFoundError:
    print("Error: occupations_en.csv file not found.")
    sys.exit(1)


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

    custom_pipeline = EntityLinker(entity_model='tabiya/bert-base-job-extract', similarity_model='all-MiniLM-L6-v2')
    extracted = custom_pipeline(job_descr)
    if not extracted:
        return jsonify({"error": "No entities extracted"}), 400

    for elem in extracted:
        if elem['type'] == "Occupation":
            new_list = []
            for esco_code in elem['retrieved']:
                row = dict_occupations[dict_occupations['code'] == esco_code]
                if not row.empty:
                    preferredLabel = row['preferredLabel'].iloc[0]
                    conceptUri = row['conceptUri'].iloc[0]
                    new_list.append({'code': esco_code, 'uri': conceptUri, 'label': preferredLabel})
            elem['retrieved'] = new_list

    return jsonify(extracted)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)