from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal
from typing_extensions import Annotated 
import pickle
import pandas as pd

#import the ml model

with open("model.pkl","rb") as f:
    model = pickle.load(f)

app = FastAPI()
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]
#pydantic model to validate incoming data

class UserInput(BaseModel):
    age:Annotated[int,Field(...,gt= 0, lt = 120, description= 'Age of the user')]
    weight:Annotated[float, Field(...,gt = 0, description = 'weight of the user in kgs')]
    height:Annotated[float, Field(...,gt = 0,lt = 2.5, description = 'eters of the user in mtrs')]
    income_lpa:Annotated[float, Field(...,gt = 0, description = 'income of the user in lakhs per annum')]
    smoker:Annotated[bool, Field(..., description = 'whether the user is a smoker or not')]
    city:Annotated[str, Field(..., description = 'city where the user lives')]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job','business_owner', 'unemployed','private_job'], Field(..., description = 'occupation of the user')]



    @computed_field
    @property
    def bmi(self) ->float:
        return self.weight/(self.height**2)


    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi>30:
            return 'high'
        elif self.smoker or self.bmi>27:
            return 'medium'
        else:
            return 'low'


    @computed_field
    @property 
    #here we are creating a computed field called age_group which is derived from the age field. This field is not stored in the database and is only used for prediction.
    #the property decorator is used to create a computed field that is derived from other fields in the model. It can also be used to create a computed field that is derived from other fields in the model and is not stored in the database.
    def age_group(self) -> str:
        if self.age<18:
            return 'young' 
        elif self.age<60:
            return 'adult'
        else:
            return 'senior'


    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
    #now to predict we will create a route that will take the user input and return the predicted insurance premium category. We will use the model that we trained earlier to make the prediction.we will use post and predict method to make the prediction. The predict method will take the user input and return the predicted insurance premium category. The user input will be validated using the pydantic model that we created earlier. The predicted insurance premium category will be returned as a response to the user.
@app.post('/predict')
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code = 200, content = {'predicted_category':prediction})

