from pydantic import BaseModel
from pydantic_core import Some

class Address(BaseModel):

    city: str
    state: str
    pin: str

class Patient(BaseModel):

    name: str
    gender: str
    age: int
    address: Address

address_dict = {'city': 'gurgaon', 'state': 'haryana', 'pin': '122001'}
#HERE ADDRESS IS A NESTED MODEL IN PATIENT MODEL. SO WE NEED TO CREATE AN INSTANCE OF ADDRESS MODEL AND PASS IT TO PATIENT MODEL.

address1 = Address(**address_dict)

patient_dict = {'name': 'aadrika', 'gender': 'female', 'age': 22, 'address': address1}

patient1 = Patient(**patient_dict)
#HERE WE ARE CREATING AN INSTANCE OF PATIENT MODEL AND PASSING THE INSTANCE OF ADDRESS MODEL TO IT.

temp = patient1.model_dump(include=)
#HERE WE ARE USING model_dump() METHOD TO DUMP THE DATA OF PATIENT MODEL. AND WE ARE USING include PARAMETER TO INCLUDE ONLY THE FIELDS THAT WE WANT TO DUMP. IN THIS CASE WE ARE INCLUDING ONLY THE NAME AND ADDRESS FIELDS.

print(type(temp))


#nested models are when we use one model as a field in another model .





















# Better organization of related data (e.g., vitals, address, insurance)

# Reusability: Use Vitals in multiple models (e.g., Patient, MedicalRecord)

# Readability: Easier for developers and API consumers to understand

# Validation: Nested models are validated automatically—no extra work needed