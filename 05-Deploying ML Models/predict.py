
import pickle
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import make_pipeline

app = FastAPI(title="Churn Prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
   result = pipeline.predict_proba(customer)[0, 1]
   return float(result)

@app.post("/predict")
def predict(customer):
    churn = predict_single(customer)
    
    return {
        
        "churn_probability": prob,
        "churn_decision": bool(churn >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)