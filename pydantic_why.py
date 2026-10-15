
from pydantic import BaseModel, EmailStr, AnyUrl , Field
from typing import List, Dict, Optional,Annotated

# EmailUrl is used to validate the email and
#  url fields in the model. It will raise a validation error if the email or url is not valid.


#Annotated is used to add metadata to the fields in the model. 
# It can be used to add description, example, and other metadata to the fields.

#field is used to add additional validation to the fields in the model.


#Basemodel is used to define the data models in pydantic. It provides automatic validation and serialization of the data.

class Patient(BaseModel):
    name: Annotated[str, Field(max_length = 50, title= 'Name  of the patient' , desription = 'Give the name of the patient in less than 50 chars , example=["Aadrika", "Ridhima"] ')]
    email: EmailStr
    linkedin_url: AnyUrl
    age: int = Field(gt = 0 , lt = 120)
    weight:Annotated[float, Field(gt = 0 , strict= True)]
    # here strict is used to fisx the data type of the field. It will raise a validation error if the data type is not correct.
    married: Annotated[bool, Field(defalut = None, description = 'Is the patient married or not?')]
    allergies: Annotated[Optional[List[str]], Field(default = None, description = 'List of allergies of the patient', max_length = 5)]
    contact_details:Dict[str,str]
#contact_details is a dictionary that contains the contact details of the patient. It can have any number of key-value pairs.


def update_patient_data(patient: Patient):

    print(patient.name)
    print(patient.age)
    print(patient.allergies)
    print(patient.married)
    print('updated')

patient_info = {'name':'aadrika', 'email':'abc@gmail.com', 'linkedin_url':'http://linkedin.com/1322', 'age': '22', 'weight': 48,'contact_details':{'phone':'2353462'}}

patient1 = Patient(**patient_info)

update_patient_data(patient1)





















































































'''
def insert_patient_data(name: str, age: int):


    if type(name) == str and type(age) == int:
        if age < 0:
            raise ValueError('Age cannot be negative')
        else:
            print(name)
            print(age)
            print('inserted into database')

    else:
        raise TypeError('Incorrect data type')
    
        


inserted_patient = insert_patient_data('aadrika','twenty one')





def update_patient_data(name,age):


    if type(name) == str and type(age) == int:
        print(name)
        print(age)
        print('updated')

    else:
        raise TypeError('Incorrect data type')
    
        




## pydantic solve the problem of type validation and data validation in fastapi. 
# It is used to define the data models and validate the incoming request data.
#  It also provides automatic documentation for the API endpoints.


'''