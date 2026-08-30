from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import __main__
from feature_engineering import FeatureEngineeringTransformer

__main__.FeatureEngineeringTransformer = FeatureEngineeringTransformer

app = FastAPI(
    title="HR Promotion Prediction API",
    description="Machine learning API for predicting employee promotions using a Soft Voting Ensemble"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

try:
    model = joblib.load('hr_promotion_pipeline.pkl')
except Exception as e:
    print(f"Lỗi: Can't found the model - {e}")


THRESHOLD = 0.60
class EmployeeData(BaseModel):
    employee_id: int


    department: Literal["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "Finance", "HR", "Legal"]
    region: Literal[
        "region_1", "region_2", "region_3", "region_4", "region_5",
        "region_6", "region_7", "region_8", "region_9", "region_10",
        "region_11", "region_12", "region_13", "region_14", "region_15",
        "region_16", "region_17", "region_18", "region_19", "region_20",
        "region_21", "region_22", "region_23", "region_24", "region_25",
        "region_26", "region_27", "region_28", "region_29", "region_30",
        "region_31", "region_32", "region_33", "region_34"
    ]


    education: Literal["Bachelor's", "Master's & above", "Below Secondary"] | None = None

    gender: Literal["Male", "Female"]
    recruitment_channel: Literal["sourcing", "other", "referred"]


    no_of_trainings: int = Field(..., ge=1)
    age: int = Field(..., ge=18)

    previous_year_rating: float | None = Field(default=None, ge=1.0, le=5.0)

    length_of_service: int = Field(..., ge=1, le=40)

    awards_won: Literal[0, 1] = Field(..., alias="awards_won?")

    avg_training_score: int = Field(..., ge=0, le=100)

@app.post("/predict")
def predict_promotion(employee: EmployeeData):
    try:

        df_input = pd.DataFrame([employee.model_dump(by_alias=True)])
        df_input['previous_year_rating'] = df_input['previous_year_rating'].astype(float)

        prob = model.predict_proba(df_input)[:, 1][0]

        is_promoted = bool(prob >= THRESHOLD)


        return {
            "status": "success",
            "prediction": {
                "probability": round(float(prob), 4),
                "strict_threshold": THRESHOLD,
                "is_promoted": is_promoted,
                "action": "Recommend for promotion" if is_promoted else "Keep under observation"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))