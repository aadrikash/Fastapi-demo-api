
def insert_patient_data(name,age):


    if type(name) == str and type(age) == int:
        print(name)
        print(age)
        print('inserted into database')

    else:
        raise TypeError('Incorrect data type')
    
        


inserted_patient = insert_patient_data('aadrika','twentyone')