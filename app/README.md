# Web Application for Job Description Analysis

This is a Flask-based API for analyzing job descriptions and predicting relevant occupations, skills, and qualifications using an entity linking model.

## Usage

First, activate the virtual environment as explained [here](../README.md#install-the-dependencies). Then, run the following command in python in the `root` directory:

### Running the API

**Run the Flask application**:

```bash
python app/server/matching.py
```

Or set the Flask application environment variable and use the Flask command:

```bash
export FLASK_APP=app/server/matching.py
flask run --host=0.0.0.0 --port=5001
```

## Example Usage

1. **Open the browser** and navigate to `http://127.0.0.1:5001/`.

2. **Paste a job description** into the provided text area.

3. **Click the "Analyze Job" button** to send the job description to the `/match` endpoint.

4. **View the results** under "Predicted Occupations," "Predicted Skills," and "Predicted Qualifications."