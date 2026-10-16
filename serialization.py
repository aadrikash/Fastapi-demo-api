from pydantic import BaseModel

class Address(BaseModel):
    
    city: str
    state: str
    pin: str

class Patient(BaseModel):
    
    name: str
    gender: str
    age: int
    address: Address

address_dict = {'city':'gurgaon','state':'haryana','pin':'122001'}
address1 = Address(**address_dict)
#here address is a nested model in patient model. so we need to create an instance of address model and pass it to patient model.
#here ** is used to unpack the dictionary and pass the values as keyword arguments 
# 
# to the constructor of the Address model.

patient_dict = {'name': 'aadrika', 'gender': 'female', 'age': 22, 'address': address1}
patient1 = Patient(**patient_dict)

temp = patient1.model_dump(exclude_unset=True)

print(temp)
print(type(temp))