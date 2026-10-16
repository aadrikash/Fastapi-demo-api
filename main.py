from fastapi import FastAPI , Path, HTTPException , Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field , computed_field
from typing_extensions import Annotated
from typing import Literal
import json


app = FastAPI()


class Patient(BaseModel):
    id:Annotated[str, Field(...,description="ID of the patient in the DB", example="P001")]
    name: Annotated[str, Field(..., description="Name of the patient", example="Aadrika")]
    city:Annotated[str, Field(..., desciption="City of the patient", example="Gurgaon")]
    age:Annotated[int, Field(..., description="Age of the patient", example=22)]
    gender:Annotated[Literal['male','female','other'], Field(..., description="Gender of the patient", example="Female")]
    height:Annotated[float, Field(..., gt = 0,description="Height of the patient in mtrs")]
    weight:Annotated[float, Field(..., gt = 0, description="Weight of the patientin kgs")]
#computed_field is a decorator that is used to create a computed field in the model. It can be used to create a computed field that is derived from other fields in the model. It can also be used to create a computed field that is derived from other fields in the model and is not stored in the database.
    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round(self.weight/(self.height**2),2)
        return bmi

    @computed_field
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return 'Underweight'
        elif self.bmi >= 18.5 and self.bmi < 25:
            return 'Normal weight'
        elif self.bmi >= 25 and self.bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

   

def load_data():
    with open('patients.json') as f:

        data = json.load(f)
    return data

def save_data(data):
    with open('patients.json','w') as f:
        json.dump(data,f)


@app.get("/")
def hello():
    return {'message': 'Patient Management System API'}

@app.get('/about')
def about():
    return {'message': 'A fully functional API to manage your patient records'}


@app.get('/view')
def view():
    data = load_data()

    return data

@app.get(('/patient/{patient_id}'))
def view_patient(patient_id: str = Path(..., description='ID of the patient in the DB' , example= 'P001')):

    data = load_data()

    if patient_id in data:
        return data[patient_id]
    else:
        raise HTTPException(status_code = 404 , detail =f'patient with ID {patient_id} not found in the database')


@app.get('/sort')
def sort_patients(sort_by : str = Query(..., description= 'Sort on the basis of height , weight or bmi'), order: str = Query('asc', description= 'sort in asc or desc order')):

    valid_fields = ['height', 'weight' , 'bmi']

    if sort_by not in valid_fields:
        raise HTTPException(status_code = 400 , detail = f'Invalid sort field. Please choose from {valid_fields}')
                                                                                                                             
    if order not in ['asc','desc']:
        raise HTTPException(status_code = 400 , detail = 'Invalid order.Please choose from asc or desc')

    data = load_data()

    sort_order = False if order == 'desc' else False

    sorted_data = sorted(data.values(), key = lambda x: x.get(sort_by,0),reverse= sort_order)



@app.post('/create')
def create_patient(patient:Patient):

    #load existing data
    data = load_data()

    #check if the patient already exist
    if patient.id in data:
        raise HTTPException(status_code = 400, detail = 'Patient already exists')

    #new patient add to database
    
    #data is in pydantic model format. we need to convert it to dictionary format before saving it to the database.

    data[patient.id] = patient.model_dump(exclude = ['id'])

    #save into json file 
    save_data(data)

    return JSONResponse(status_code = 201 , content = {'message':f'patient created successfully with ID {patient.id}'})

        
        